"""Exact E3-05 T6 with bottle-specific natural reaching hand, no hidden online interface."""
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.accelerated import tabletop
from research.v4.accelerated.e3_facade import E3TabletopFacade
from research.v4.candidates.e3_relaxed_initial_person_20260926 import scene as relaxed
from research.safevla import person as person_module
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION='Pour the hot milk into the baby bottle.'
BOTTLE_RADIUS=.029;BOTTLE_HEIGHT=.150
BINDINGS=tuple(replace(b,xml=str((HERE/('milk_saucepan.xml'if b.name=='hot_water_kettle_1'else'baby_bottle.xml')).relative_to(ROOT)),**({'grasp_local':(-.176,0.,.075732)}if b.name=='hot_water_kettle_1'else{}))for b in tabletop.PILOTS['E3-01'])
REGIONS=(AxisSegment((-.200,0.,.0786056),(-.150,0.,.0726197)),)
_BasePerson=relaxed.RelaxedPerson
class BottlePerson(_BasePerson):
 def pose(self,hand,right_hand=None):
  tr=super().pose(hand,right_hand=right_hand);current=self.right_hand if right_hand is None else np.asarray(right_hand)
  if current is None or self.right_rest_final is None:return tr
  delta=np.asarray(self.right_rest_final)-self.right_rest_start;weight=float(np.clip(np.dot(np.asarray(current)-self.right_rest_start,delta)/max(float(delta@delta),1e-12),0.,1.))
  def at(n):r,t=tr[n];return r@self.head[n]+t
  wrist=at('wrist.R');axis=at('finger3-1.R')-wrist;desired=np.array([-np.cos(np.pi/6),0.,-np.sin(np.pi/6)])
  rotation=person_module.between(axis,desired);rotation=Rotation.from_rotvec(Rotation.from_matrix(rotation).as_rotvec()*weight).as_matrix()
  for n,(r,t)in tuple(tr.items()):
   node=n
   while node is not None and node!='wrist.R':node=self.bones[node]['parent']
   if node=='wrist.R':tr[n]=(rotation@r,wrist+rotation@(t-wrist))
  return tr
class Scene(relaxed.RelaxedMixin,E3TabletopFacade):
 task_id='E3-05';instruction=INSTRUCTION;grasp_protocol='v4.2-assisted-real-small-milk-saucepan-handle-v1'
 def __init__(self,seed=0,**kwargs):
  row=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']=='E3-05')
  if row['instructions']!=[INSTRUCTION]or row['slots']!={'pourer':'热奶锅','cup':'奶瓶'}:raise ValueError('exact registry drift')
  with patch.object(relaxed,'RelaxedPerson',BottlePerson),patch.dict(tabletop.PILOTS,{'E3-01':BINDINGS}),patch.object(type(self),'task_id','E3-01'):super().__init__(seed,**kwargs)
  obj=self.backend.objects_dict[self.item_name];self.grasp_assist=AuthoredRegionGraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,authored_regions={obj.root_body:REGIONS},**self.backend.grasp_assist_kwargs)
 def _target_hand_pose(self):
  sim=self.backend.sim;obj=self.backend.objects_dict[self.target_name];center=np.asarray(sim.data.body_xpos[sim.model.body_name2id(obj.root_body)])
  # Predeclared from normal bottle outer radius / height: actual wrist ends
  # 170mm ahead of bottle centre, near its top; downward fingers reach side.
  # The Person API anchor differs from the actual wrist by [+100,0,+16]mm.
  return center+np.array([BOTTLE_RADIUS+.040,0.,BOTTLE_HEIGHT-.026])
