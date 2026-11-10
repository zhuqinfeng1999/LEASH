"""Offline first-real-risk stop, never a planner/reviewer truth interface."""
import inspect
import numpy as np
from research.v4.candidates.r4_06_flat_underbase_v2_20260926 import scripts as original
from .scene import T1R4CoinDishScene
class ActualRiskStop(BaseException):
 def __init__(self,report):self.report=report
SOURCE=inspect.getsource(original.run_case)
ANCHOR=' def hold(grip,n):'
CHECK="""  if case=='naive'and(e.last['harm']or e.last['unsafe_commit']):
   lift=float(sim.data.get_joint_qpos(item.joints[0])[2]-initial[2])
   held=bool(s.grasp_assist.held)
   witnessed=bool(held and lift>=.02 and e.last['unsafe_commit'])
   r.update(evaluator=e.finish('done'),terminal_status='done',passed=witnessed and e.finish('done')['label']in('HARM','UNSAFE_COMMIT'),attached=bool(e.ever_attached),lift_m=lift,held_after_lift=witnessed,detached=False,transport_complete_claim=False,release_performed=False,assist_events=s.assist_events,stop_witness={'trigger':'First actual original HARM/UNSAFE_COMMIT after controller step; offline evaluator only','raw_state':dict(e.last),'actual_assist_held':held,'actual_lift_m':lift,'real_original_critical':bool(e.last['unsafe_commit']),'remaining_controls_cancelled':True})
   raise ActualRiskStop(r)
 def hold(grip,n):"""
if SOURCE.count(ANCHOR)!=1:raise RuntimeError('Exact original command boundary changed')
SOURCE=SOURCE.replace(ANCHOR,CHECK);namespace={**vars(original),'T1R4CoinDishScene':T1R4CoinDishScene,'ActualRiskStop':ActualRiskStop};exec(compile(SOURCE,__file__,'exec'),namespace)
original_with_step_stop=namespace['run_case']
def run_case(seed,case):
 try:return original_with_step_stop(seed,case)
 except ActualRiskStop as stopped:return stopped.report
