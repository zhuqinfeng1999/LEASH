"""Fresh two-view E2 geometry, from public RGB-D/SAM only."""
import time
import numpy as np
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicBox,PublicWorkspace
from .name_binding import PROMPTS

class PublicWorkspaceProvider:
 def __init__(self,*,capture,adapter,sam3_client,log,seconds):
  self.capture=capture;self.adapter=adapter;self.sam3=sam3_client;self.log=log;self.seconds=seconds
 def __call__(self):
  start=time.perf_counter();packet=self.capture();self.seconds['public_capture']=self.seconds.get('public_capture',0)+time.perf_counter()-start
  cameras=packet['cameras']
  if set(cameras)!={'agentview','robot0_eye_in_hand'}:raise ValueError('E2 requires exactly two public views')
  for cam in cameras.values():
   if np.asarray(cam['rgb']).shape!=(480,640,3)or np.asarray(cam['depth_m']).shape!=(480,640)or cam['frame_id']!=packet['frame_id']:raise ValueError('E2 public shape/frame alignment invalid')
  boxes=[]
  for name,prompt in PROMPTS.items():
   views=[]
   for view in ('agentview','robot0_eye_in_hand'):
    start=time.perf_counter()
    try:
     box,details=sam_public_box(name=name,prompt=prompt,camera=cameras[view],sam3_client=self.sam3,min_score=.2)
     views.append(box);self.log.append(dict(frame_id=packet['frame_id'],view=view,**details))
    except Exception as exc:self.log.append(dict(frame_id=packet['frame_id'],view=view,name=name,prompt=prompt,error=f'{type(exc).__name__}: {exc}'))
    finally:self.seconds['sam3']=self.seconds.get('sam3',0)+time.perf_counter()-start
   if views:
    lo=np.min([x.lower for x in views],axis=0);hi=np.max([x.upper for x in views],axis=0)
    boxes.append(PublicBox(name,tuple(map(float,lo)),tuple(map(float,hi))))
  eef=tuple(float(x)for x in packet['robot_state']['eef_pos'])
  # No physical action occurs between snapshot and evidence on owner thread.
  evidence=self.adapter.capture_commit_evidence()
  return PublicWorkspace(evidence,eef,tuple(boxes))

def locate_vla_target(prompt,public):
 text=prompt.lower()
 for name,phrases in (('box',('box','bin','container')),('dictionary',('dictionary','book'))):
  box=public.box(name)
  if box is not None and any(word in text for word in phrases):return name,tuple((a+b)/2 for a,b in zip(box.lower,box.upper))
 return None
