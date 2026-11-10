"""Opt-in accounting at the existing SDK turn submission, not a new stream."""
from contextlib import contextmanager
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from .progress import available, claim, update

ENV_VALUE = 'transport-workspace-dev1-20261005'


def record_request(*, retry=False):
    if os.environ.get('SAFEVLA_TRANSPORT_WORKSPACE_BUDGET') != ENV_VALUE:
        return
    if os.environ.get('SAFEVLA_EXECUTION_PROFILE') == 'confidence_formal_20261008':
        # Current formal authorization replaces development-only admission.
        # This is still the same Thread.turn and collector, not another call.
        root=Path(__file__).resolve().parents[4]
        plan_path=Path(os.environ.get('SAFEVLA_FORMAL_PLAN',str(root/'research/v4/results/formal_confidence_20261008/plan.json'))).resolve()
        if not plan_path.is_relative_to(root/'research/v4/results') or plan_path.name!='plan.json':
            raise RuntimeError('Formal plan must be a registered project results plan')
        plan=json.loads(plan_path.read_text())
        output=Path(os.environ['SAFEVLA_RUN_OUTPUT']).resolve()
        rows=[r for r in plan['episodes'] if r['run_id']==output.name
              and r['status']=='running' and r['phase']=='formal']
        if len(rows)!=1 or (root/rows[0]['output']).resolve()!=output:
            raise RuntimeError('Current active formal request registration required')
        from research.v4.bridge.formal_scope_controls import in_scope
        if not in_scope(plan,rows[0]):
            raise RuntimeError('Formal planner/method pair paused by user')
        if plan['authorization']['state'] not in ('RUNNING','DRAINING_FOR_FORMAL_ADMISSION_REPAIR'):
            raise RuntimeError('Formal request admission paused')
        with (output/'formal_request_admissions.jsonl').open('a') as f:
            f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                retry=retry,scope='admission before existing SDK turn; actual turn IDs counted separately'))+'\n')
        return
    if retry:
        def change(data):
            if not available(data):
                raise RuntimeError('Development STOP or fixed deadline before service retry')
            data['used']['service_retry_requests'] += 1
        update(change)
    else:
        claim('business_model_requests')


@contextmanager
def reviewer_submission(thread, *, retry=False):
    if os.environ.get('SAFEVLA_TRANSPORT_WORKSPACE_BUDGET') != ENV_VALUE:
        yield
        return
    cls = type(thread)
    original = cls.turn
    def turn(current,input,**kwargs):
        if current is thread:
            record_request(retry=retry)
        return original(current,input,**kwargs)
    with patch.object(cls,'turn',turn):
        yield
