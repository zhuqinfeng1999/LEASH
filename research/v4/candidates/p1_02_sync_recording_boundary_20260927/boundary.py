"""Opt-in external synchronous SDK transport substitute; no planner replacement."""
from contextlib import contextmanager
from unittest.mock import patch
import openai_codex.api as api
from openai_codex.client import CodexClient
from openai_codex.models import Notification
from openai_codex.generated.v2_all import ItemCompletedNotification, TurnCompletedNotification

class UnexpectedExternalIO(RuntimeError): pass

class RecordedSyncClient(CodexClient):
    """Retain SDK serialization/router/handles; substitute process/RPC boundary only."""
    def __init__(self, config=None):
        super().__init__(config=config)
        self.records=[];self.started=False;self.closed=False
    def start(self):
        if self.started or self.closed: raise UnexpectedExternalIO('duplicate/closed start')
        self.started=True
        self.records.append({'method':'recorded_start','config_overrides':list(self.config.config_overrides),
            'cwd':self.config.cwd,'experimental_api':self.config.experimental_api,
            'environment_values_recorded':False})
    def close(self): self.closed=True
    def notify(self, method, params):
        if method!='initialized' or params is not None: raise UnexpectedExternalIO('unlisted notification')
        self.records.append({'method':method,'params':params})
    def request(self, method, params, *, response_model):
        if not self.started or self.closed: raise UnexpectedExternalIO('outside recording lifecycle')
        expected=['initialize','thread/start','turn/start']
        seen=[r['method'] for r in self.records if r['method'] in expected]
        if len(seen)>=3 or method!=expected[len(seen)]: raise UnexpectedExternalIO('unlisted/out-of-order RPC')
        self.records.append({'method':method,'params':params,'response_class':response_model.__name__})
        if method=='initialize':
            value={'userAgent':'recorded-qa/1.0.0','serverInfo':{'name':'recorded-qa','version':'1.0.0'}}
        elif method=='thread/start':
            thread={'id':'recorded-sync-thread','cliVersion':'recorded-qa','createdAt':0,'updatedAt':0,
                'cwd':params['cwd'],'ephemeral':True,'modelProvider':'recorded-qa','preview':'',
                'sessionId':'recorded-qa-session','source':'cli','status':{'type':'idle'},'turns':[]}
            value={'thread':thread,'approvalPolicy':params['approvalPolicy'],
                'approvalsReviewer':params.get('approvalsReviewer','user'),'cwd':params['cwd'],
                'model':params['model'],'modelProvider':'recorded-qa','sandbox':{'type':'dangerFullAccess'}}
        else:
            turn={'id':'recorded-sync-turn','items':[],'status':'completed'}
            item={'id':'recorded-finish','type':'mcpToolCall','server':'rpent','tool':'finish',
                'arguments':{'status':'done','message':'Recorded transport finish; no actual tool execution or safety decision.'},
                'status':'completed','result':{'content':[]}}
            events=[Notification('item/completed',ItemCompletedNotification.model_validate({
                'threadId':'recorded-sync-thread','turnId':'recorded-sync-turn','completedAtMs':0,'item':item})),
                Notification('turn/completed',TurnCompletedNotification.model_validate({
                    'threadId':'recorded-sync-thread','turn':turn}))]
            for event in events: self._router.route_notification(event)
            value={'turn':turn}
        return response_model.model_validate(value)

@contextmanager
def recorded_boundary():
    """Install only on explicit scope; never patch Codex/Thread/Turn/Planner methods."""
    clients=[]
    def factory(config=None):
        c=RecordedSyncClient(config);clients.append(c);return c
    # Guard accidental runtime entry even if an unanticipated SDK path is used.
    with patch.object(api,'CodexClient',factory), \
         patch.object(CodexClient,'start',side_effect=UnexpectedExternalIO('live runtime forbidden')):
        yield clients
