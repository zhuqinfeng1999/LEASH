"""One initial horizontal source; actual common physics/assist unchanged."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.p4_02_straightener_heatmat_20260926 import facade as original
from research.v4.candidates.p4_v42_t4_tabletop_20260926 import facade as common
HERE=Path(__file__).resolve().parent
BASE_ENV=common._P4Env
class FlatCeramicEnv(BASE_ENV):
 def _load_model(self):
  super()._load_model();tray=self.model.worldbody.find("body[@name='heat_resistant_left_tray']");board=self.model.worldbody.find("body[@name='tabletop_ironing_board']");tr=Rotation.from_quat(np.roll(np.fromstring(tray.get('quat'),sep=' '),-1)).as_matrix();br=Rotation.from_quat(np.roll(np.fromstring(board.get('quat'),sep=' '),-1)).as_matrix();rel=tr.T@br;stand=tray.find("body[@name='p402_horizontal_ceramic_cradle']")
  for g in stand.findall('geom'):
   pos=np.fromstring(g.get('pos'),sep=' ');name=g.get('name');x={'p402_ceramic_base':0.,'p402_ceramic_hinge_support':-.105,'p402_ceramic_head_support':.105}[name];v=rel@np.array([x,0.,0.]);pos[:2]=v[:2];g.set('pos',' '.join(map(str,pos)));g.set('quat',' '.join(map(str,np.roll(Rotation.from_matrix(rel).as_quat(),1))))
 def _reset_internal(self):
  super()._reset_internal();m,d=self.sim.model,self.sim.data;b=m.body_name2id('tabletop_ironing_board');t=m.body_name2id('heat_resistant_left_tray');br=np.asarray(d.body_xmat[b]).reshape(3,3);p=np.asarray(d.body_xpos[t]).copy();p[2]=.939;R=br.copy();obj=self.objects_dict['p4_iron_1'];d.set_joint_qpos(obj.joints[0],np.r_[p,np.roll(Rotation.from_matrix(R).as_quat(),1)]);d.set_joint_qvel(obj.joints[0],np.zeros(6));self.sim.forward()
class Scene(original.StraightenerScene):
 def __init__(self,*args,**kwargs):
  sample=common.sample_slot
  def initialized(slot,*a,**kw):
   xyz,yaw=sample(slot,*a,**kw)
   return(np.asarray(xyz).copy()+np.array([.160,0,0])if slot=='R'else xyz),yaw
  with patch.object(original,'HERE',HERE),patch.object(common,'_P4Env',FlatCeramicEnv),patch.object(common,'sample_slot',initialized):super().__init__(*args,**kwargs)
