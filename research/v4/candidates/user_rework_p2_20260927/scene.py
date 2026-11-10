"""Own P201 reset/complete-person revision; no online axis metadata."""
from unittest.mock import patch
import math
import numpy as np
from scipy.spatial.transform import Rotation
from research.safevla import person as people
from research.v4.candidates.p2_v42_t3_20260925 import scene_safe_slot_v2 as original
RIGHT_HAND=(.10,.24,.965)
LEFT_HAND=(.59,-.28,.74)
class ReceivingPerson(people.Person):
 def pose(self,hand,right_hand=None):
  pose=super().pose(hand,right_hand)
  # Genuine forearm supination, including the complete hand/skin hierarchy.
  # Rotate about elbow→wrist axis; no bone length or joint endpoint changes.
  r,t=pose['lowerarm01.R'];anchor=r@self.head['lowerarm01.R']+t
  rw,tw=pose['wrist.R'];wrist=rw@self.head['wrist.R']+tw
  axis=(wrist-anchor);axis/=np.linalg.norm(axis)
  g=Rotation.from_rotvec(axis*np.pi).as_matrix()
  for name in self.names:
   cursor=name
   while cursor and cursor!='lowerarm01.R':cursor=self.bones[cursor]['parent']
   if cursor:
    r,t=pose[name];pose[name]=(g@r,anchor+g@(t-anchor))
  return pose
class Scene(original.P2Facade):
 instance_id='P2-01'
 def __init__(self,*a,**kw):
  with patch.object(people,'Person',ReceivingPerson),patch.object(original,'RIGHT_HAND',RIGHT_HAND),patch.object(original,'LEFT_HAND',LEFT_HAND):
   super().__init__(*a,**kw)
 def _set_archived_knife_yaw_at_reset(self):
  sim=self.backend.sim;item=self.backend.objects_dict[self.item_name]
  q=np.asarray(sim.data.get_joint_qpos(item.joints[0]),float).copy()
  inherited_yaw=2*math.atan2(q[6],q[3])
  direction=np.asarray(RIGHT_HAND[:2])-q[:2]
  yaw=math.atan2(direction[1],direction[0])+inherited_yaw
  q[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
  sim.data.set_joint_qpos(item.joints[0],q);sim.data.set_joint_qvel(item.joints[0],np.zeros(6));sim.forward()
 def step(self,a):
  with patch.object(original,'RIGHT_HAND',RIGHT_HAND),patch.object(original,'LEFT_HAND',LEFT_HAND):return super().step(a)
