"""Task-local identical full skeleton IK, only smooth right elbow pole selectable."""
import numpy as np
from scipy.spatial.transform import Rotation
from research.safevla.person import between
def pose(self, hand, right_hand=None):
 transforms={};hand=np.asarray(hand)
 targets={'L':hand+np.array([.10,0,.016])}
 right_hand=self.right_hand if right_hand is None else right_hand
 if right_hand is not None:targets['R']=np.asarray(right_hand)+np.array([.10,0,.016])
 elbow=None
 def visit(n):
  nonlocal elbow
  if n in transforms:return transforms[n]
  p=self.bones[n]['parent'];r,t=visit(p) if p else (self.rest_rotation,self.rest_translation)
  h=self.head[n];world=r@h+t;direction=None
  if n=='spine02':r=Rotation.from_euler('y',-20,degrees=True).as_matrix()@r
  if n.startswith('upperleg01.'):direction=np.array([-.97,0,-.10])
  if n.startswith('lowerleg01.'):direction=np.array([.06,0,-1.])
  if n.startswith('foot.'):direction=np.array([-1.,0,0])
  if n=='upperarm01.R' and 'R' not in targets:direction=np.array([.15,0,-1.])
  if n=='lowerarm01.R' and 'R' not in targets:direction=np.array([-.35,-.1,-1.])
  side=n[-1]
  if n.startswith('upperarm01.') and side in targets:
   wrist_target=targets[side]
   l1=np.linalg.norm(self.head['lowerarm01.'+side]-h);l2=np.linalg.norm(self.head['wrist.'+side]-self.head['lowerarm01.'+side])
   delta=wrist_target-world;distance=np.linalg.norm(delta);u=delta/distance
   reach=min(distance,l1+l2-.001);a=(l1*l1-l2*l2+reach*reach)/(2*reach)
   perpendicular=np.array([0,-1.,0.]) if side=='L' else np.asarray(getattr(self,'_right_elbow_pole',[1.,.5,-.7]),dtype=float).copy();perpendicular-=u*np.dot(perpendicular,u);perpendicular/=np.linalg.norm(perpendicular)
   elbow=world+u*a+perpendicular*np.sqrt(max(0,l1*l1-a*a));direction=elbow-world
   restdir=self.head['lowerarm01.'+side]-h
   r=between(r@restdir,direction)@r;direction=None
  if n.startswith('lowerarm01.') and side in targets:
   restdir=self.head['wrist.'+side]-h
   r=between(r@restdir,targets[side]-world)@r
  if n.startswith('wrist.') and side in targets:
   # Align metacarpal axis toward the robot and dorsal side upward.
   f=self.head['finger3-1.'+side]-h;f/=np.linalg.norm(f)
   across=self.head['finger5-1.'+side]-self.head['finger2-1.'+side];across-=f*np.dot(across,f);across/=np.linalg.norm(across)
   normal=np.cross(f,across)
   sign=1 if side=='L' else -1
   desired=np.column_stack(([-1.,0,0],[0,-sign,0],[0,0,sign]))
   r=desired@np.column_stack((f,across,normal)).T
  if n=='finger1-1.R' and 'R' in targets:
   # Thumb abduction keeps the thumb clear while fingertips contact the rim.
   r=Rotation.from_rotvec([-.55,0,0]).as_matrix()@r
  if direction is not None:r=between(r@(self.tail[n]-h),direction)@r
  t=world-r@h;transforms[n]=(r,t);return r,t
 for n in self.names:visit(n)
 return transforms
