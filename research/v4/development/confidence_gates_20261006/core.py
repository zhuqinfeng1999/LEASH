"""Confidence gate decision core. No geometry, simulator or backend imports."""
from dataclasses import dataclass
import hashlib
import json
import math

LABELS={'G1':('PROCEED','REFUSE'),'G2':('APPROVE','DENY'),'G3':('PASS','BLOCK')}
@dataclass(frozen=True)
class Thresholds:
    tau_task:float=0
    tau_action:float=0
    tau_commit:float=0
    def __post_init__(self):
        for value in (self.tau_task,self.tau_action,self.tau_commit):
            if isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Threshold must be finite probability')
    def for_gate(self,gate):return dict(G1=self.tau_task,G2=self.tau_action,G3=self.tau_commit)[gate]

def candidate_hash(candidate):
    return hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def valid_probability(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=1

class GateCore:
    """Backend score is normalized to [0,1]; certainty never affects permission."""
    def __init__(self,backend,thresholds=Thresholds()):
        self.backend=backend;self.thresholds=thresholds;self.records=[];self.cache={};self.refused=False
    def score(self,gate,evidence,*,candidate,epoch):
        digest=candidate_hash(candidate);key=(gate,epoch,digest)
        if key in self.cache:return self.cache[key]
        attempts=[];answer=None
        for attempt in range(2): # one technical retry; valid denials never retried
            try:raw=self.backend.score(gate,evidence)
            except Exception as exc:raw={'p_safe':None,'error':f'{type(exc).__name__}: {exc}'}
            attempts.append(raw)
            if isinstance(raw,dict) and valid_probability(raw.get('p_safe')):
                answer=raw;break
        allowed=answer is not None and answer['p_safe']>=self.thresholds.for_gate(gate)
        row=dict(gate=gate,epoch=epoch,candidate_hash=digest,allowed=allowed,
                 decision=LABELS[gate][0 if allowed else 1],
                 status='scored' if answer is not None else 'system_failure',
                 p_safe=answer['p_safe'] if answer is not None else None,
                 attempts=attempts,threshold=self.thresholds.for_gate(gate))
        self.records.append(row);self.cache[key]=row
        if gate=='G1' and not allowed:self.refused=True
        return row
    def may_call(self,name):return not self.refused or name=='finish'
    def action_route(self,*,holding,open_only):
        return 'G3' if open_only and holding in ('HELD','UNKNOWN') else 'G2'
    def end_primitive(self):
        # Approval/cache cannot outlive the primitive; denied entries are retained.
        self.cache={k:v for k,v in self.cache.items() if not v['allowed']}
