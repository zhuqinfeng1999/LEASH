"""Root-only finite own trio; original branches/evaluator/main driver are reused."""
import inspect,hashlib,types
from pathlib import Path
from research.v4.candidates.r3_03_t1_derived_20260926 import scripts as original
from research.v4.candidates.r3_03_own_front_carrier_20260928.initial_wrist import InitialWristMirrorFrontThermosScene
from research.v4.candidates.r3_04_source_axis_low_actual_scripts_20260928 import entry as runner
from .expert import run_naive,journal,initial
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
class SelectedOwnThermosScene(InitialWristMirrorFrontThermosScene):
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id,root_initial_wrist_approved=True)

def selected():
    src=inspect.getsource(original.run_case)
    start=src.index('        else:\n            # The camera-ready start');end=src.index("        details['assist_events']",start)
    prefix,suffix=src[:start],src[end:]
    src=prefix+"        else:\n            run_naive(scene,evaluator,command,details,initial_z)\n"+suffix
    if not src.startswith(prefix):raise ValueError('Own original correct/ability branch changed')
    old='            return evaluator.observe(scene)'
    if src.count(old)!=1:raise ValueError('Own evaluator single observe seam drift')
    src=src.replace(old,'            state=evaluator.observe(scene)\n            journal(scene,action,evaluator)\n            return state')
    old='        sim = scene.backend.sim'
    if src.count(old)!=1:raise ValueError('Own initial environment seam drift')
    src=src.replace(old,old+'\n        initial(scene,details)')
    old='        scene.close()';assert src.count(old)==1
    src=src.replace(old,"        details['physical_trace']=getattr(scene,'offline_trace',[])\n        details['true_bind']=getattr(scene,'offline_bind',None)\n        details['selected_script_version']='R303-own-horizontal-initial-mirror-v1'\n"+old)
    ns=dict(original.run_case.__globals__,R3TabletopFacade=SelectedOwnThermosScene,run_naive=run_naive,journal=journal,initial=initial)
    exec(compile(src,str(H/'entry.py')+'::own-original-case-ordinary-horizontal-driver','exec'),ns)
    return ns['run_case'],src,{'own_original_correct_and_empty16_branches_preserved':True,'original_exception_finish_priority_preserved':True,'original_global_sample_wall_limits':[1200,90.],'source_guard_before_construct':True,'default_shared_sources_untouched':True}

def admission():
    return types.FunctionType(runner.admission.__code__,dict(runner.admission.__globals__,H=H,ROOT=ROOT),runner.admission.__name__)()

def main():
    ns=dict(runner.main.__globals__,H=H,ROOT=ROOT,original=original,selected=selected,admission=admission,HorizontalSideFoilScene=SelectedOwnThermosScene)
    types.FunctionType(runner.main.__code__,ns,runner.main.__name__)()
if __name__=='__main__':main()
