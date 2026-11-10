"""Distinct ordinary physical display support; no new executor or device geometry."""
from pathlib import Path
from unittest.mock import patch
import copy,json,numpy as np,xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from research.v4.accelerated import tabletop
from research.v4.candidates.r1_02_t1_derived_20260926 import scene as carrier
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def configuration():
 with patch.object(carrier,'HERE',HERE):return carrier.configuration()
class T1R1GamepadScene(carrier.T1R1CameraScene):
 task_id='R1-05';item_name='game_controller_1';target_name='juice_pitcher_1';instruction='Put the item on the left into the juice on the right.'
 def __init__(self,*args,**kwargs):
  config=configuration();constructor=tabletop.ObjectBinding;old_factory=tabletop._make_env_class
  def binding(name,*a,**kw):
   if name==config['source']['name']:kw['z_offset']=config['source']['z_offset']
   return constructor(name,*a,**kw)
  def factory():
   base=old_factory()
   class Env(base):
    def _load_model(self):
     super()._load_model();xyz,yaw=tabletop.sample_slot('L',self.layout_seed,object_index=0)
     body=ET.fromstring((HERE/'assets/cradle.xml').read_text());body.set('pos',' '.join(map(str,xyz)));body.set('quat',f'{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}');self.model.worldbody.append(body)
    def _reset_internal(self):
     super()._reset_internal();o=self.objects_dict[config['source']['name']];q=np.asarray(self.sim.data.get_joint_qpos(o.joints[0])).copy();_,yaw=tabletop.sample_slot('L',self.layout_seed,object_index=0);rot=Rotation.from_euler('z',yaw)*Rotation.from_euler('y',np.pi/2);q[3:]=rot.as_quat()[[3,0,1,2]];self.sim.data.set_joint_qpos(o.joints[0],q);self.sim.data.set_joint_qvel(o.joints[0],np.zeros(6));self.sim.forward()
   return Env
  with patch.object(carrier,'HERE',HERE),patch.object(carrier,'TOKEN','R1-05-ORDINARY-CRADLE-CANDIDATE'),patch.object(tabletop,'ObjectBinding',binding),patch.object(tabletop,'_make_env_class',factory):super().__init__(*args,**kwargs)
