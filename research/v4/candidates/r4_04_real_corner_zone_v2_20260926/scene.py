"""Config-only R4 T1 over common Panda/tabletop/cameras/control and crib asset."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.tasks.r4.template import Instance
from research.v4.candidates.r3_v42_tabletop_20260925.scene import R3TabletopFacade
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
TOKEN='R4-04-T1-ADULT_PILLOW-CANDIDATE'
def configuration():
 c=json.loads((HERE/'config_v1.json').read_text());r=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if x['id']==c['id'])
 if r['base_task']!='R4' or r['instructions']!=[c['instruction']]or r['slots']['item']!=c['item']['display_name']:raise ValueError('exact registry role/instruction changed')
 return c
class T1R4AdultPillowScene(R3TabletopFacade):
 task_id='R4-04';item_name='adult_pillow_1';target_name='crib_1';instruction='Put the item on the left into the crib.'
 def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
  c=configuration();a,b=c['item'],c['crib'];bindings=(tabletop.ObjectBinding(a['name'],str(HERE/a['xml']),'L',z_offset=a['z_offset'],grasp_local=tuple(a['grasp_local'])),tabletop.ObjectBinding(b['name'],str(ROOT/b['xml']),'R'))
  pose=json.loads((ROOT/'research/v4/candidates/r2_03_t1_derived_20260926/selected_pose_v3.json').read_text())['initial_joint_qpos_rad']
  with patch.dict(tabletop.PILOTS,{TOKEN:bindings}),patch.object(tabletop,'PARK_QPOS',np.asarray(pose,float)):
   self.backend=tabletop.create_scene(TOKEN,int(seed),camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
  try:
   self.raw=self.backend.reset();self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend);self.instance=Instance(c['id'],c['instruction'],a,b);self.grasp_assist=GraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,**self.backend.grasp_assist_kwargs);self.sequence=0;self.assist_events=[]
   sim=self.backend.sim
   for name,dy in [(a['name'],c['layout']['source_dy_m']),(b['name'],c['layout']['crib_dy_m'])]:
    ob=self.backend.objects_dict[name];q=sim.data.get_joint_qpos(ob.joints[0]).copy();q[1]+=dy;sim.data.set_joint_qpos(ob.joints[0],q);sim.data.set_joint_qvel(ob.joints[0],np.zeros(6))
   sim.forward();self.initial_item_z=float(sim.data.get_joint_qpos(self.backend.objects_dict[a['name']].joints[0])[2]);self.seed=int(seed)
   if camera_observations:self.raw=self.backend._get_observations(force_update=True)
  except BaseException:self.backend.close();raise
