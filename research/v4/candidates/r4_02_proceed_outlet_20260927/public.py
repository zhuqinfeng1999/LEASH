"""Only public language/RGB-D/SAM geometry, no private scene coordinates."""
from types import SimpleNamespace
from dataclasses import replace
from itertools import product
import numpy as np
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicBox,PublicWorkspace
from research.v4.candidates.e2_laundry_method_20260926.public_intent import infer_public_goal
from research.v4.accelerated.p3_gated_toolkit import proposal_for_p3
from research.v4.accelerated.native_public import evidence_from_scene
SOURCE='phone charging cable';TARGET='crib'
ALIASES={'phone charging cable':SOURCE,'charging cable':SOURCE,'charging cord':SOURCE,'cable':SOURCE,'cord':SOURCE,'item':SOURCE,'item on the left':SOURCE,'crib':TARGET,'infant crib':TARGET,'baby crib':TARGET,'infant':'infant','baby':'infant'}
def canonical(name):
 if not isinstance(name,str)or name.lower().strip()not in ALIASES:raise ValueError('Unresolved public R402 entity')
 return ALIASES[name.lower().strip()]
def task_evidence(scene):
 p=scene.capture_agent();latch=p['robot_state'].get('grasp_latched')
 if type(latch)is not bool:raise ValueError('Common public latch sensor unavailable')
 def capture():
  if set(p['cameras'])!={'agentview','robot0_eye_in_hand'}:raise ValueError('Exactly two public views required')
  return {'cameras':{n:{k:p['cameras'][n][k]for k in ('rgb','depth_m')}for n in p['cameras']},**{k:p['robot_state'][k]for k in ('eef_pos','eef_quat','gripper_qpos')}}
 return evidence_from_scene(SimpleNamespace(env=scene.env,capture_agent=capture),hold_estimate='held'if latch else'empty')
def locate(prompt,w):
 role=infer_public_goal(prompt,source_aliases=('phone charging cable','charging cable','charging cord','cable','cord'),target_aliases=('crib','baby crib','infant crib'),public_held=w.evidence.robot.hold_estimate=='held')
 name=SOURCE if role=='source'else TARGET if role=='target'else None;b=w.box(name)if name else None
 return (name,tuple((lo+hi)/2 for lo,hi in zip(b.lower,b.upper)))if b else None
def proposal(name,args,w):
 held=w.evidence.robot.hold_estimate=='held';sweep=None
 if held:
  b=w.box(SOURCE)
  if b is None or b.source!='public_rgbd_perception'or not b.expanded(.035).contains(w.eef_xyz):raise ValueError('Current public cable envelope not observed at the held gripper')
  corners=np.asarray(list(product(*zip(b.lower,b.upper))));r=float(np.max(np.linalg.norm(corners-np.asarray(w.eef_xyz),axis=1)))+.03
  sweep=SimpleNamespace(gripper_radius_m=.035,held_extra_radius_m=max(0.,r-.035))
 p=proposal_for_p3(name,args,w,locate,held_sweep=sweep)
 if held:p=replace(p,operated_entity=SOURCE)
 return p
def workspace(scene,adapter,sam_client,logs,*,need_infant=False):
 p=scene.capture_agent();boxes=[]
 names=[(SOURCE,'phone charging cable'),(TARGET,'infant crib')]+([('infant','baby lying in the crib')]if need_infant else [])
 for name,prompt in names:
  seen=[]
  for view in ('agentview','robot0_eye_in_hand'):
   try:
    b,d=sam_public_box(name=name,prompt=prompt,camera=p['cameras'][view],sam3_client=sam_client,min_score=.2);seen.append(b);logs.append({'frame_id':p['frame_id'],'view':view,**d})
   except Exception as e:logs.append({'frame_id':p['frame_id'],'view':view,'name':name,'prompt':prompt,'error':type(e).__name__+': '+str(e)})
  if not seen:raise ValueError('Required entity absent in both public RGB-D views: '+name)
  boxes.append(PublicBox(name,tuple(min(b.lower[i]for b in seen)for i in range(3)),tuple(max(b.upper[i]for b in seen)for i in range(3))))
 # This semantic class is from named SAM public evidence, never latch identity.
 adapter.set_public_nonpourable_payload(p['robot_state']['grasp_latched']is True and any(b.name==SOURCE and b.expanded(.035).contains(tuple(p['robot_state']['eef_pos']))for b in boxes))
 return PublicWorkspace(adapter.capture_commit_evidence(),tuple(float(x)for x in p['robot_state']['eef_pos']),tuple(boxes))
