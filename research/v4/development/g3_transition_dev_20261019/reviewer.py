"""Spawn-safe Luna worker; only the G3 scope contract changes."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from research.v4.development.confidence_gate_dev_20261006 import backend
from research.v4.development.confidence_gates_20261006.config import prompt as original_prompt
from .scope import VERSION


def scoped_prompt(gate,*args,**kwargs):
    if gate=='G3':return Path(__file__).with_name('prompt_G3.txt').read_text()
    return original_prompt(gate,*args,**kwargs)


class TransitionLunaWorker(backend.LunaWorker):
    def review(self,gate,packet):
        from openai_codex.client import CodexClient
        request=CodexClient._request_raw
        path=Path(self.config.log_path).with_name('reviewer_sdk_configuration.jsonl')
        def recorded(client,method,params=None):
            params=params or {}
            if method in ('thread/start','turn/start'):
                row=dict(utc=datetime.now(timezone.utc).isoformat(),method=method,gate='G'+str(gate),
                    fields={k:params[k] for k in ('model','threadId','effort','serviceTier') if k in params})
                instruction=params.get('baseInstructions')
                if instruction is not None:
                    expected=scoped_prompt('G'+str(gate))+'\n\nShared safety policy:\n'+self.config.policy_text
                    if instruction!=expected:raise ValueError('Actual reviewer system prompt differs from the scoped version')
                    row.update(base_instructions_sha256=sha256(instruction.encode()).hexdigest(),
                               prompt_version=VERSION if gate==3 else 'unchanged-original')
                with path.open('a') as f:f.write(json.dumps(row)+'\n')
            return request(client,method,params)
        with patch.object(backend,'prompt',scoped_prompt),patch.object(CodexClient,'_request_raw',recorded):
            return super().review(gate,packet)


class TransitionScoreSession(backend.ScoreSession):
    def _log(self,row):
        if row.get('gate')=='G3':row=dict(row,prompt_version=VERSION)
        return super()._log(row)

    def call(self,gate,*args,**kwargs):
        result=super().call(gate,*args,**kwargs)
        if gate=='G3':result['prompt_version']=VERSION
        return result
