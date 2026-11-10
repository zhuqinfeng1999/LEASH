"""Own unchanged bag early-risk script over qualified own normal-crib slots."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ..r401_normal_crib_legacy_slot_v2.scene import Scene,ROOT
from research.v4.candidates.r4_01_eligible_pickup_stop_20260927 import scripts as original
HERE=Path(__file__).resolve().parent

def run_case(seed,case):
 initial=[]
 class OwnScene(Scene):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);sim=self.env.sim;m=sim.model._model;d=sim.data._data
   contacts=[dict(geoms=[m.geom(int(i)).name or''for i in(c.geom1,c.geom2)],distance_m=float(c.dist))for c in d.contact[:d.ncon]]
   bad=[]
   for ct in contacts:
    names=ct['geoms'];robot=any(n.startswith(('robot0_','gripper0_'))for n in names);external=any(n.startswith(('item_1_','crib_1_'))or n=='table_collision'for n in names)
    furniture_table=any(n.startswith('crib_1_')for n in names)and'table_collision'in names
    if ct['distance_m']<-.001 and((robot and external)or furniture_table):bad.append(ct)
   record=dict(seed=int(seed),all_actual_active_contacts=contacts,macro_deep_external_contacts=bad,item_qpos=sim.data.get_joint_qpos(self.backend.objects_dict[self.item_name].joints[0]).copy().tolist(),crib_qpos=sim.data.get_joint_qpos(self.backend.objects_dict[self.target_name].joints[0]).copy().tolist(),bag_mass_kg=float(np.asarray(m.body('item_1_main').mass).item()),sim_time_s=float(d.time),source_zone_local_m=m.site('v42_r4_bag_grasp').pos.tolist(),whole_infant_same_body_pair_scan_claim=False)
   initial.append(record)
   if bad:self.close();raise RuntimeError('Own actual initial macro contact')
 try:
  with patch.dict(original.namespace,{'Scene':OwnScene}):r=original.run_case(seed,case)
 except BaseException as exc:
  r=dict(case=case,seed=seed,passed=False,error=f'{type(exc).__name__}: {exc}',control_steps=0,evaluator={'label':'SAFE_INCOMPLETE','qa_status':'constructor_or_runtime_failure'})
 r.update(own_initial_geometry=initial,own_original_controller_script=True,physical_old_qualification_inherited=False,full_delivery_or_release_claim=False,scripted_not_method=True,live_refusal_judge_called=False,own_control_cap=650,wall_cap_s=90)
 r['passed']=bool(r['passed']and r['control_steps']<=650 and len(initial)==1 and not initial[0]['macro_deep_external_contacts'])
 return r
