"""Original powered straightener/vanity/towel/mat, common T4 executor."""
from pathlib import Path
from unittest.mock import patch
import json,numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.p4_v42_t4_tabletop_20260926 import facade as base
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];TASK='P4-02-T4-CANDIDATE'
PARK=np.asarray(json.loads((HERE.parent/'p4_public_reset_20260926/selected_pose_v2.json').read_text())['initial_joint_qpos_rad'])
# Real upper cold-handle coordinates, obtained from the authored8deg housing.
A=np.deg2rad(8.);R=np.array([[np.cos(A),0,-np.sin(A)],[0,1,0],[np.sin(A),0,np.cos(A)]])
C=np.array([.005,0,.030]);REGIONS=(AxisSegment(tuple(C+R@[-.105,0,.004]),tuple(C+R@[-.050,0,.004])),)
GRASP=tuple(C+R@[-.0775,0,.004])
BINDING=tabletop.ObjectBinding('p4_iron_1',str(HERE/'assets/iron.xml'),'L',z_offset=.020,grasp_local=GRASP)
class StraightenerScene(base.P4TabletopFacade):
 task_id='P4-02';instruction='Put the hair straightener on the vanity.'
 public_item_description='powered rose-gold hair straightener';public_target_description='silicone heat mat on vanity'
 grasp_protocol='v4.2-assisted-authored-region-straightener-cold-handle-v1'
 def __init__(self,*args,**kwargs):
  row=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if x['id']==self.task_id)
  if row['instructions']!=[self.instruction]:raise ValueError('registry drift')
  with patch.multiple(base,HERE=HERE,TASK=TASK),patch.dict(tabletop.PILOTS,{TASK:(BINDING,)}),patch.object(tabletop,'PARK_QPOS',PARK):super().__init__(*args,**kwargs)
  obj=self.backend.objects_dict[self.item_name]
  self.grasp_assist=AuthoredRegionGraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,authored_regions={obj.root_body:REGIONS},**self.backend.grasp_assist_kwargs)
  self.visual_render_changes=[]
  for n in obj.contact_geoms:
   gid=self.backend.sim.model.geom_name2id(n);self.visual_render_changes.append(dict(geom=n,old_group=int(self.backend.sim.model.geom_group[gid]),new_group=1));self.backend.sim.model.geom_group[gid]=1
  if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
