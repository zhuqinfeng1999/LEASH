"""Config T1 over common carrier; real articulated top drawer is open at reset."""
from pathlib import Path
from unittest.mock import patch
import json,math,numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r2_02_t1_derived_20260926 import scene as existing
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
POSE_PATH=ROOT/'research/v4/candidates/r2_03_t1_derived_20260926/selected_pose_v3.json'
PARK_QPOS=np.asarray(json.loads(POSE_PATH.read_text())['initial_joint_qpos_rad'],float)

class _DrawerEnv(tabletop._make_env_class()):
 def _load_model(self):
  super()._load_model()
  # Same thin correction already used by the accepted R3 microwave: the
  # generic reset expects the injected free joint before articulated joints.
  obj=self.objects_dict['paper_wooden_cabinet_1'];free=[n for n in obj._joints if n=='joint0']
  if len(free)!=1:raise RuntimeError('cabinet free joint missing')
  obj._joints=free+[n for n in obj._joints if n!=free[0]]

def _create_scene(task_id,seed,**kwargs):return _DrawerEnv(task_id,seed,**kwargs)

class T1R2DrawerScene(existing.T1R2DerivedScene):
 task_id='R2-05';item_name='lit_candle_1';target_name='paper_wooden_cabinet_1';instruction='Put the item on the left into the open drawer on the right.'
 def __init__(self,*args,**kwargs):
  with patch.object(existing,'HERE',HERE),patch.object(existing,'TOKEN','R2-05-T1-DERIVED-CANDIDATE'),patch.object(tabletop,'PARK_QPOS',PARK_QPOS),patch.object(tabletop,'create_scene',_create_scene):super().__init__(*args,**kwargs)
  sim=self.backend.sim;obj=self.backend.objects_dict[self.target_name];q=np.asarray(sim.data.get_joint_qpos(obj.joints[0])).copy()
  from research.v4.tasks.r2.evaluator import _yaw
  yaw=_yaw(q)+self.instance.target['initial_yaw_offset_rad'];q[3:]=[math.cos(yaw/2),0,0,math.sin(yaw/2)];sim.data.set_joint_qpos(obj.joints[0],q)
  sim.data.set_joint_qpos(self.target_name+'_'+self.instance.target['open_drawer_joint_suffix'],self.instance.target['initial_drawer_open_m'])
  sim.forward();self.raw=self.backend._get_observations(force_update=True)
  self.initial_reset_revision='r2-05-reuse-r2-03-reset35-higher1.44-v1'
