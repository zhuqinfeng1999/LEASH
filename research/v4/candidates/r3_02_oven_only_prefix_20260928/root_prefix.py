"""Thin own R302 bindings into existing root-only prefix recorder, not a new executor."""
import inspect,hashlib,json,sys
from pathlib import Path
import numpy as np
from unittest.mock import patch
from research.v4.candidates.r3_05_oven_only_prefix_20260928 import root_prefix as recorder
from research.v4.candidates.r3_02_cc0_soup_config_20260926 import qualify_cpu as own
from research.v4.candidates.r3_02_cc0_soup_config_20260926.scene import SoupCanScene
from research.v4.candidates.r3_03_t1_derived_20260926 import scripts as original
from .scene import OvenOnlySoupScene
H=Path(__file__).resolve().parent
ORIGINAL_INITIAL_NONDEVICE=recorder.initial_nondevice
INITIAL_DEVICE_METADATA=[]

def initial_nondevice(scene):
 result=ORIGINAL_INITIAL_NONDEVICE(scene)
 sim=scene.backend.sim;m,d=sim.model._model,sim.data._data
 item=scene.backend.objects_dict[scene.item_name];oven=scene.backend.objects_dict[scene.oven_name]
 INITIAL_DEVICE_METADATA.append({
  "selected_class":type(scene).__module__+"."+type(scene).__name__,
  "actual_oven_joint_qpos":np.asarray(sim.data.get_joint_qpos(oven.joints[0])).tolist(),
  "actual_source_joint_qpos":np.asarray(sim.data.get_joint_qpos(item.joints[0])).tolist(),
  "actual_fixture_eq_data":m.eq_data[m.equality("r3_microwave_fixture").id].tolist(),
  "actual_fixture_active":bool(d.eq_active[m.equality("r3_microwave_fixture").id]),
  "actual_all_qvel":d.qvel.tolist(),"actual_sequence":scene.sequence,"sim_time_s":float(d.time),
  "model_dt_s":float(m.opt.timestep),"offline_only":True})
 return result

# Existing exact recorder's contact payload name is its only fork-bowl-specific
# field. Rebind that exact function to this own can prefix; all comparisons,
# forces, geometry and >1mm non-grasp first-failure policy remain byte-identical.
CONTACT_SOURCE=inspect.getsource(recorder.contact_rows)
if CONTACT_SOURCE.count("'fork_bowl_1_'")!=1:raise ValueError('Contact recorder payload seam changed')
CONTACT_SELECTED=CONTACT_SOURCE.replace("'fork_bowl_1_'","'steel_thermos_1_'")
_ns=dict(recorder.contact_rows.__globals__)
exec(compile(CONTACT_SELECTED,__file__+'::own-public-name-offline-contact-binding','exec'),_ns)
contact_rows=_ns['contact_rows']

def selected_prefix(Scene=OvenOnlySoupScene):
 with patch.multiple(original,Evaluator=own.bound_evaluator,JUDGE=own.JUDGE,REASON=own.REASON),patch.multiple(recorder,original=original,H=H):
  return recorder.compiled_prefix(Scene)

def main():
 # Actual existing main/finite allocator/recorder executes only after root launch.
 # This scope retains original R302 evaluator identity and restores every donor
 # binding even after an intentional first-failure exception.
 with patch.multiple(original,Evaluator=own.bound_evaluator,JUDGE=own.JUDGE,REASON=own.REASON),patch.multiple(recorder,original=original,H=H,T1R3ForkBowlScene=SoupCanScene,OvenOnlyPrefixScene=OvenOnlySoupScene,contact_rows=contact_rows,initial_nondevice=initial_nondevice):
  INITIAL_DEVICE_METADATA.clear()
  try:return recorder.main()
  finally:
   if "--output"in sys.argv:
    out=Path(sys.argv[sys.argv.index("--output")+1]).resolve()
    if out.is_dir():(out/"actual_initial_device_source_metadata.json").write_text(json.dumps(INITIAL_DEVICE_METADATA,indent=2)+"\n")
if __name__=='__main__':main()
