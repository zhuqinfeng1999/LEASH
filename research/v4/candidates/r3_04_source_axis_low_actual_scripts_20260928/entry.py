"""Root-only actual scripts: compile original case runner with one offline naive branch."""
import argparse,json,inspect,hashlib,time,ast
from pathlib import Path
from research.v4.candidates.r3_04_foil_sandwich_20260926 import scripts as original
from research.v4.candidates.r3_04_horizontal_side_grasp_copy_20260928.scene import HorizontalSideFoilScene
from .expert import run_naive,journal
H=Path(__file__).resolve().parent;ROOT=H.parents[3]

def selected():
 src=inspect.getsource(original.run_case)
 start=src.index('        else:\n            # The camera-ready start')
 end=src.index('        details[\'assist_events\'] = scene.assist_events',start)
 prefix=src[:start];suffix=src[end:]
 src=prefix+"        else:\n            run_naive(scene,evaluator,command,details,initial_z)\n"+suffix
 old='            return evaluator.observe(scene)'
 if src.count(old)!=1:raise ValueError('Original single-observe command seam drift')
 src=src.replace(old,'            state = evaluator.observe(scene)\n            journal(scene, action, evaluator)\n            return state')
 marker='        scene.close()'
 if src.count(marker)!=1:raise ValueError('Original final evidence seam drift')
 src=src.replace(marker,"        details['physical_trace'] = getattr(scene,'offline_trace',[])\n        details['true_bind'] = getattr(scene,'offline_bind',None)\n        details['selected_script_version'] = 'R304-source-axis-low-actual-v1'\n"+marker)
 ns=dict(original.run_case.__globals__);ns.update(R3TabletopFacade=HorizontalSideFoilScene,run_naive=run_naive,journal=journal)
 exec(compile(src,str(H/'entry.py')+'::original-case-and-actual-OSC-expert','exec'),ns)
 return ns['run_case'],src,{'original_correct_and_capability_branches_preserved':True,'original_exception_finish_priority_preserved':True,'original_global_sample_wall_limits':[original.MAX_STEPS,original.MAX_WALL_S],'source_guard_before_construct':True,'default_shared_sources_untouched':True}

def admission():
 packet=json.loads((H/'source_freeze.json').read_text());bad=[]
 for name,sha in packet.items():
  p=Path(name);p=p if p.is_absolute()else ROOT/p
  if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:bad.append(name)
 if bad:raise ValueError('Frozen source admission rejected '+str(bad))
 return len(packet)

def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--seed',type=int,default=0);p.add_argument('--output');a=p.parse_args();N=admission();f,src,contract=selected()
 if a.preflight:
  tree=ast.parse((H/'expert.py').read_text());writes=[]
  for node in ast.walk(tree):
   if isinstance(node,(ast.Assign,ast.AugAssign)):
    targets=node.targets if isinstance(node,ast.Assign)else[node.target]
    for target in targets:
     if 'qpos'in ast.unparse(target)or'eq_data'in ast.unparse(target):writes.append(ast.unparse(target))
  if writes:raise ValueError('Forbidden live state writes '+str(writes))
  print(json.dumps(dict(contract,pins=N,selected_factory=f.__globals__['R3TabletopFacade'].__name__,generated_source_sha256=hashlib.sha256(src.encode()).hexdigest(),live_state_writes=writes,Scene_controls_GPU_models=0)));return
 if a.seed!=0 or not a.output:raise ValueError('This first actual authorization entry is seed0 only/fresh output required')
 out=Path(a.output).resolve()
 if out.exists():raise ValueError('Fresh output only; never overwrite or retry')
 out.mkdir(parents=True);(out/'selected_script.py.txt').write_text(src);receipt={'source_sha256':json.loads((H/'source_freeze.json').read_text()),'case_order':['correct','naive','capability_failure'],'cases':[],'first_failure_stop':True,'natural_method_or_native_qualification':False,'GPU_models_services':0,'selected_factory':HorizontalSideFoilScene.__name__};start=time.monotonic()
 try:
  for case in receipt['case_order']:
   result=f(0,case);(out/(case+'_seed0.json')).write_text(json.dumps(result,indent=2)+'\n');receipt['cases'].append({'case':case,'passed':result['passed'],'label':result['evaluator']['label'],'controls':result['control_steps'],'error':result['error']})
   if not result['passed']:break
 except Exception as exc:
  receipt['startup_or_system_error'] = type(exc).__name__+': '+str(exc)
  raise
 finally:
  receipt['elapsed_s']=time.monotonic()-start;receipt['source_drift']=[n for n,sha in receipt['source_sha256'].items()if hashlib.sha256((Path(n)if Path(n).is_absolute()else ROOT/n).read_bytes()).hexdigest()!=sha];(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({k:receipt[k]for k in ['cases','elapsed_s','source_drift']}))
if __name__=='__main__':main()
