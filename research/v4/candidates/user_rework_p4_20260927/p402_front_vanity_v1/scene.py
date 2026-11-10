"""One front-tray/rear-mirror tabletop dressing organizer."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.user_rework_p4_20260927.p402_vanity_v1 import scene as v1
from research.v4.candidates.user_rework_p4_20260927.p402_vanity_v2 import scene as v2
HERE=Path(__file__).resolve().parent
class FrontVanityEnv(v1.VanityEnv):
 def _load_model(self):
  super()._load_model()
  tray=self.model.worldbody.find("body[@name='heat_resistant_left_tray']");board=self.model.worldbody.find("body[@name='tabletop_ironing_board']")
  for body,delta in [(tray,(.21,.06,0.)),(board,(.02,-.04,0.))]:
   p=np.fromstring(body.get('pos'),sep=' ')+delta;body.set('pos',' '.join(map(str,p)))
  tr=Rotation.from_quat(np.roll(np.fromstring(tray.get('quat'),sep=' '),-1)).as_matrix();br=Rotation.from_quat(np.roll(np.fromstring(board.get('quat'),sep=' '),-1)).as_matrix()@Rotation.from_euler('z',90,degrees=True).as_matrix();board.set('quat',' '.join(map(str,np.roll(Rotation.from_matrix(br).as_quat(),1))))
  toolR=br@Rotation.from_euler('z',180,degrees=True).as_matrix();rel=tr.T@toolR
  for g in tray.find(".//body[@name='p402_horizontal_ceramic_cradle']").findall('geom'):
   p=np.fromstring(g.get('pos'),sep=' ');x={'p402_ceramic_base':0.,'p402_ceramic_hinge_support':-.105,'p402_ceramic_head_support':.105}[g.get('name')];p[:2]=(rel@np.array([x,0.,0.]))[:2];g.set('pos',' '.join(map(str,p)));g.set('quat',' '.join(map(str,np.roll(Rotation.from_matrix(rel).as_quat(),1))))
 def _reset_internal(self):
  super()._reset_internal();m,d=self.sim.model,self.sim.data;b=m.body_name2id('tabletop_ironing_board');R=np.asarray(d.body_xmat[b]).reshape(3,3)@Rotation.from_euler('z',180,degrees=True).as_matrix();obj=self.objects_dict['p4_iron_1'];q=np.asarray(d.get_joint_qpos(obj.joints[0])).copy();q[3:]=np.roll(Rotation.from_matrix(R).as_quat(),1);d.set_joint_qpos(obj.joints[0],q);d.set_joint_qvel(obj.joints[0],np.zeros(6));self.sim.forward()
class Scene(v2.Scene):
 layout_version='p402-normal-rear-mirror-front-worktop-short-held-pose-transfer-v1'
 def __init__(self,*args,**kwargs):
  with patch.object(v2,'HERE',HERE),patch.object(v1,'VanityEnv',FrontVanityEnv):super().__init__(*args,**kwargs)
