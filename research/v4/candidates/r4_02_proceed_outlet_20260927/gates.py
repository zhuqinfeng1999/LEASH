"""Thin primitive boundary over existing ActionGate and common commit outlet."""
from dataclasses import replace,asdict
from research.v4.accelerated.p3_gated_adapter import P3GatedAdapter
from research.v4.accelerated.p3_gated_toolkit import ROBOT_ACTIONS,OBSERVATION_OR_FILE_TOOLS
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import ActionGate
from research.v3.bridge.b0 import _tool_result
from . import public
class ReviewerRelay:
 def __init__(self):self.bound=None
 def bind(self,reviewer):
  if self.bound is not None:raise RuntimeError('Reviewer relay already bound')
  self.bound=reviewer
 def review(self,**kw):
  if self.bound is None:raise RuntimeError('Independent reviewer unavailable')
  return self.bound.review(**kw)
class Adapter(P3GatedAdapter):
 def __init__(self,scene,*,instruction,on_sim_step=None):
  self.relay=ReviewerRelay();self._public_nonpourable=False
  super().__init__(scene,instruction=instruction,reviewer=self.relay,default_commit_conditions=('default home-safety policy',),on_sim_step=on_sim_step)
 def bind_reviewer(self,reviewer):self.relay.bind(reviewer)
 def set_public_nonpourable_payload(self,observed):
  if type(observed)is not bool:raise ValueError('Public semantic observation must be Boolean')
  self._public_nonpourable=observed
 def _state_from_packet(self,packet):
  s=super()._state_from_packet(packet)
  if s.hold_estimate=='empty':self._public_nonpourable=False
  return replace(s,held_object_tiltable=False)if s.hold_estimate=='held'and self._public_nonpourable else s
def bind_task(decision):
 c=decision.constraints
 return replace(decision,constraints=replace(c,allowed_operated_entities=frozenset(public.canonical(x)for x in c.allowed_operated_entities),forbidden_entities=frozenset(public.canonical(x)for x in c.forbidden_entities),orientation_rules=tuple(replace(r,operated_entity=public.canonical(r.operated_entity))for r in c.orientation_rules)))
class MotionReviewer:
 def __init__(self,reviewer):self.reviewer=reviewer
 def review_motion(self,**kw):
  r=self.reviewer.review_motion(**kw)
  return {**r,'forbidden_entities':[public.canonical(x)for x in r.get('forbidden_entities',[])]}
class Toolkit:
 def __init__(self,base,adapter,*,decision,reviewer,workspace,round_id):
  self.base=base;self.adapter=adapter;self.workspace=workspace;self.proposals=[];self.perception_log=[];self.pending=None
  self.gate=ActionGate(round_id=round_id,task=bind_task(decision),snapshot=lambda:self.pending or self.workspace(),reviewer=MotionReviewer(reviewer),finish=lambda status,message:self.base.execute_tool('finish',{'status':status,'message':message}),timeout_s=180)
 def __getattr__(self,name):return getattr(self.base,name)
 def get_tools_spec(self):return self.base.get_tools_spec()
 def execute_tool(self,name,args):
  if name=='finish':
   r=self.base.execute_tool(name,args);self.gate.end_round();self.adapter.set_commit_conditions(None);return r
  if name in OBSERVATION_OR_FILE_TOOLS:return self.base.execute_tool(name,args)
  if name not in ROBOT_ACTIONS:return _tool_result(name,{'error':'Unregistered R402 primitive','code':'action_denied','gate':2})
  try:
   self.pending=self.workspace();p=public.proposal(name,args,self.pending);d=self.gate.check(p)
  except Exception as e:
   d=self.gate._deny('Public R402 proposal unavailable: '+type(e).__name__+': '+str(e),'Refresh current public RGB-D source/crib evidence',reviewer_called=False,system_failure=True);p=None
  finally:self.pending=None
  row={'tool':name,'proposal':asdict(p)if p else None,'decision':d.result,'reason':d.reason,'constraint':d.constraint};self.proposals.append(row)
  if d.result!='APPROVE' or not self.gate.is_current_lease(d.lease):
   self.adapter.set_commit_conditions(None);return _tool_result(name,{'error':d.reason,'code':'action_denied','gate':2,'constraint':d.constraint})
  self.adapter.set_commit_conditions(d.lease.commit_conditions);self.adapter.last_outcome=None
  try:r=self.base.execute_tool(name,args)
  finally:self.adapter.set_commit_conditions(None)
  o=self.adapter.last_outcome
  if o is not None and o.status=='intercepted':
   row['commit_outcome']=asdict(o);return _tool_result(name,{'error':o.reason,'code':'commit_intercepted','gate':3,'discarded_commands':o.discarded})
  if 'error'not in(getattr(r,'result',{})or{}):self.gate.on_primitive_completed(p)
  return r
 def cancel_active_and_wait(self):return self.base.cancel_active_and_wait()
 def close(self):return self.base.close()
