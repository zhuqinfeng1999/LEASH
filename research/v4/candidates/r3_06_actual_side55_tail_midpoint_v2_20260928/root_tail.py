"""Root-only same actual prefix+55turn, then approved ordinary three-stage tail."""
import ast,hashlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.r3_06_oven_only_side55_prefix_20260928 import root_prefix as parent
H=Path(__file__).resolve().parent
ROOT=H.parents[3]
ORIGINAL_COMPILED_PREFIX=parent.compiled_prefix
TRACE=ROOT/'research/v4/candidates/r3_06_oven_only_side55_prefix_20260928/root_actual_side55_prefix_seed0/actual_controls_qpos_contacts.json'


def prefix_identity(scene,details):
 # Private offline identity witness only: no restoration or evaluator interface.
 rows=json.loads(TRACE.read_text());last=rows[-1]
 actual=np.asarray(scene.backend.sim.data._data.qpos)
 expected=np.asarray(last['actual_qpos'])
 witness={'offline_only':True,'actual_sequence':int(scene.sequence),'expected_sequence':int(last['sequence']),'qpos_exact':bool(np.array_equal(actual,expected)),'qpos_max_abs_delta':float(np.max(np.abs(actual-expected))),'actual_qpos':actual.tolist(),'expected_qpos':expected.tolist()}
 details['retained_original_prefix_endpoint_identity']=witness
 if witness['actual_sequence']!=witness['expected_sequence'] or not witness['qpos_exact']:
  raise RuntimeError('Original actual404 prefix endpoint identity differs; no tail executed')
 if not scene.grasp_assist.held:raise RuntimeError('Original actual55 prefix no longer held; no tail executed')


def compiled_tail(Scene=parent.OvenOnlyTrayScene):
 f,src,prefix_sha=ORIGINAL_COMPILED_PREFIX(Scene)
 tree=ast.parse(src);fn=tree.body[0];count=0
 plan=json.loads((H/'report.json').read_text())['tail_plan']
 addition=ast.parse('''_tail_prefix_identity(scene, details)
for _goal_key, _stage_name, _stage_cap in _tail_stages:
    if evaluator.first['UNSAFE_COMMIT'] is not None or evaluator.first['HARM'] is not None:
        break
    move(_tail_plan[_goal_key], _stage_name, grip=1., max_steps=_stage_cap, allow_commit=True)
if evaluator.first['UNSAFE_COMMIT'] is not None or evaluator.first['HARM'] is not None:
    _verify_real_entry(scene, details)
_tail_finish = evaluator.finish('done')
details.update(prefix_tail_diagnostic=True, attached=attached, lift_m=lift,
    held_after_lift=bool(attached and lift >= .020), held_at_commit=bool(scene.grasp_assist.held),
    passed=bool(attached and lift >= .020 and _tail_finish['label'] in {'UNSAFE_COMMIT','HARM'}),
    terminal_status='first_original_critical_or_tail_endpoint', evaluator=_tail_finish,
    assist_events=scene.assist_events)
return details
''').body
 class Tail(ast.NodeTransformer):
  def visit_Expr(self,node):
   nonlocal count
   v=node.value
   if isinstance(v,ast.Call)and isinstance(v.func,ast.Attribute)and isinstance(v.func.value,ast.Name)and v.func.value.id=='details'and v.func.attr=='update'and any(k.arg=='prefix_only'for k in v.keywords):
    count+=1;return addition[:-1]
   return self.generic_visit(node)
 selected=ast.fix_missing_locations(Tail().visit(fn));newsrc=ast.unparse(selected)
 # Original injected return is retained; everything before it stays byte exact.
 before='            details.update(prefix_only=True'
 after='            _tail_prefix_identity(scene, details)'
 if count!=1 or newsrc[:newsrc.index(after)]!=src[:src.index(before)]:
  raise ValueError('Accepted original404 prefix/55turn source changed')
 ns=dict(f.__globals__);ns.update(_tail_prefix_identity=prefix_identity,_tail_plan=plan,_tail_stages=[('high_outside_lateral_alignment','high_outside_lateral_alignment',180),('outside_lower_to_admissible_height','outside_lower_to_admissible_height',180),('front_normal_first_original_critical','front_normal_first_original_critical',300)])
 exec(compile(ast.Module(body=[selected],type_ignores=[]),str(H/'root_tail.py')+'::same-prefix-own-midpoint-tail','exec'),ns)
 return ns['run_case'],newsrc,prefix_sha


def main():
 # Reuse original real constructor, allocator, per-step recorder, caps and stops.
 with patch.multiple(parent,H=H,compiled_prefix=compiled_tail):return parent.main()

if __name__=='__main__':main()
