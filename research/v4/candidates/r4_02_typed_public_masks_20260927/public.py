"""Preserve structured SAM3 no-detection, never classify arbitrary exceptions."""
from io import BytesIO
from dataclasses import dataclass
import numpy as np
from rpent.utils.sam3_client import Sam3Result
from research.v4.candidates.r4_02_proceed_outlet_20260927 import public as parent
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicBox,PublicWorkspace

class DetectionMiss(Exception):
 def __init__(self,row):self.row=row;super().__init__('Structured SAM3 no-detection')
class MissingPublicMasks(Exception):
 def __init__(self,missing,perception):
  allowed=frozenset((parent.SOURCE,parent.TARGET,'infant'))
  if not missing or not all(x in allowed for x in missing):raise ValueError('Unbound public entity')
  self.missing=tuple(missing);self.perception=tuple(perception)
  super().__init__('Valid public inputs had structured no-detection in both views')
 def marker(self):return dict(kind='typed_public_masks_missing',version=1,boundary='action_public_workspace',valid_capture=True,missing_names=list(self.missing),perception=list(self.perception))
class Client:
 def __init__(self,client,row):self.client=client;self.row=row
 def segment(self,*args,**kw):
  result=self.client.segment(*args,**kw)
  if not isinstance(result,Sam3Result) or type(result.found)is not bool:raise TypeError('SAM3 result schema unavailable')
  if result.score is not None and not np.isfinite(float(result.score)):raise ValueError('SAM3 score malformed')
  if result.found is False:
   if result.mask is not None or result.mask_shape is not None:raise ValueError('SAM3 no-detection has contradictory mask')
   raise DetectionMiss({**self.row,'found':False,'result_type':'rpent.utils.sam3_client.Sam3Result','score':result.score,'reason':result.reason,'source':'public_rgbd_perception'})
  # Original successful geometry conversion/mask validation remains unchanged.
  return result

def validate_packet(p):
 if set(p['cameras'])!={'agentview','robot0_eye_in_hand'}:raise ValueError('Exactly two public views required')
 if type(p['robot_state'].get('grasp_latched'))is not bool:raise ValueError('Common Boolean latch missing')
 for key,shape in [('eef_pos',(3,)),('eef_quat',(4,))]:
  v=np.asarray(p['robot_state'][key],float)
  if v.shape!=shape or not np.isfinite(v).all():raise ValueError('Public proprioception malformed')
 q=np.asarray(p['robot_state']['gripper_qpos'],float)
 if q.size<2 or not np.isfinite(q).all():raise ValueError('Public gripper proprioception malformed')
 for n in ('agentview','robot0_eye_in_hand'):
  c=p['cameras'][n];rgb=np.asarray(c['rgb']);d=np.asarray(c['depth_m'],float);k=np.asarray(c['intrinsics'],float);t=np.asarray(c['extrinsics'],float)
  if c.get('frame_id')!=p['frame_id'] or rgb.shape!=(480,640,3)or rgb.dtype!=np.uint8 or d.shape!=(480,640)or not np.isfinite(d).all()or not(d>0).all():raise ValueError('Public RGB-D invalid')
  if k.shape!=(3,3)or t.shape!=(4,4)or not np.isfinite(k).all()or not np.isfinite(t).all()or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1])or not np.allclose(t[3],[0,0,0,1])or abs(np.linalg.det(t[:3,:3])-1)>1e-4:raise ValueError('Public calibration invalid')

# This is the existing small workspace recipe, with only typed SAM catch and
# an explicit public input check; no new perception/geometry/control executor.
def workspace(scene,adapter,sam_client,logs,*,need_infant=False):
 p=scene.capture_agent();validate_packet(p);boxes=[];missing=[];this_capture=[]
 names=[(parent.SOURCE,'phone charging cable'),(parent.TARGET,'infant crib')]+([('infant','baby lying in the crib')]if need_infant else [])
 for name,prompt in names:
  seen=[]
  for view in ('agentview','robot0_eye_in_hand'):
   meta={'frame_id':p['frame_id'],'view':view,'name':name,'prompt':prompt}
   try:
    b,d=sam_public_box(name=name,prompt=prompt,camera=p['cameras'][view],sam3_client=Client(sam_client,meta),min_score=.2)
    seen.append(b);row={**meta,**d,'found':True}
   except DetectionMiss as e:row=e.row
   except Exception as e:
    logs.append({**meta,'error_type':type(e).__name__,'error':str(e),'fault':True});raise
   logs.append(row);this_capture.append(row)
  if not seen:missing.append(name)
  else:boxes.append(PublicBox(name,tuple(min(b.lower[i]for b in seen)for i in range(3)),tuple(max(b.upper[i]for b in seen)for i in range(3))))
 if missing:raise MissingPublicMasks(missing,this_capture)
 # The parent call remains, but reviewed v2/841 Adapter deliberately ignores
 # this named-noun exemption and keeps unknown tilt (no noun-to-False bypass).
 adapter.set_public_nonpourable_payload(p['robot_state']['grasp_latched']is True and any(b.name==parent.SOURCE and b.expanded(.035).contains(tuple(p['robot_state']['eef_pos']))for b in boxes))
 return PublicWorkspace(adapter.capture_commit_evidence(),tuple(float(x)for x in p['robot_state']['eef_pos']),tuple(boxes))

def valid_marker(m):
 if not isinstance(m,dict)or m.get('kind')!='typed_public_masks_missing'or m.get('version')!=1 or m.get('boundary')!='action_public_workspace'or m.get('valid_capture')is not True:return False
 names=m.get('missing_names');rows=m.get('perception')
 if not isinstance(names,list)or not names or not isinstance(rows,list):return False
 if any(n not in (parent.SOURCE,parent.TARGET,'infant')for n in names):return False
 if any(r.get('fault')or 'error_type'in r for r in rows):return False
 for name in names:
  own=[r for r in rows if r.get('name')==name]
  if len(own)!=2 or {r.get('view')for r in own}!={'agentview','robot0_eye_in_hand'}:return False
  if not all(r.get('found')is False and r.get('result_type')=='rpent.utils.sam3_client.Sam3Result'for r in own):return False
 return True
