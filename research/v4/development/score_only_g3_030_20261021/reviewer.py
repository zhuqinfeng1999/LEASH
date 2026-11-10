"""Spawn-safe score-only SDK reviewer using the original lifecycle."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from research.v4.development.confidence_gate_dev_20261006 import backend
from .profile import VERSION, SCHEMA, normalize_luna, score_only_prompt


class ScoreOnlyLunaWorker(backend.LunaWorker):
    def review(self, gate, packet):
        from openai_codex.client import CodexClient
        request = CodexClient._request_raw
        path = Path(self.config.log_path).with_name('reviewer_sdk_configuration.jsonl')

        def recorded(client, method, params=None):
            params = params or {}
            if method in ('thread/start', 'turn/start'):
                row = dict(utc=datetime.now(timezone.utc).isoformat(), method=method,
                    gate='G' + str(gate), prompt_version=VERSION,
                    fields={k: params[k] for k in ('model', 'threadId', 'effort', 'serviceTier', 'outputSchema') if k in params})
                instruction = params.get('baseInstructions')
                if instruction is not None:
                    expected = score_only_prompt('G' + str(gate)) + '\n\nShared safety policy:\n' + self.config.policy_text
                    if instruction != expected:
                        raise ValueError('Actual reviewer instructions differ from the frozen score-only prompt')
                    row['base_instructions_sha256'] = sha256(instruction.encode()).hexdigest()
                if 'outputSchema' in params and params['outputSchema'] != SCHEMA:
                    raise ValueError('Actual reviewer schema differs from score-only schema')
                with path.open('a') as f:
                    f.write(json.dumps(row) + '\n')
            return request(client, method, params)

        with patch.object(backend, 'prompt', score_only_prompt), patch.object(CodexClient, '_request_raw', recorded):
            return super().review(gate, packet)

    def _score_turn(self, gate, packet):
        with patch.object(backend, 'SCHEMA', SCHEMA):
            return super()._score_turn(gate, packet)


class ScoreOnlySession(backend.ScoreSession):
    def _log(self, row):
        return super()._log(dict(row, prompt_version=VERSION, reviewer_output_contract=VERSION))

    def call(self, gate, *args, **kwargs):
        with patch.object(backend, 'normalize_luna', normalize_luna):
            result = super().call(gate, *args, **kwargs)
        result['prompt_version'] = VERSION
        return result
