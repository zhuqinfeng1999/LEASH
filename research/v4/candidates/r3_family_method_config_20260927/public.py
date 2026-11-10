"""Own public-RGBD role grounding, no private scene/evaluator interface."""
from dataclasses import replace
from types import SimpleNamespace
from itertools import product
import numpy as np
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.accelerated.p3_gated_toolkit import proposal_for_p3
from research.v4.candidates.e2_laundry_method_20260926.public_intent import infer_public_goal
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicWorkspace

def proposal(b,name,args,workspace,locate):
 held=workspace.evidence.robot.hold_estimate=='held';sweep=None
 if held:
  box=workspace.box(b.source)
  if box is None or box.source!='public_rgbd_perception':raise ValueError('own current public payload envelope missing')
  corners=np.array(list(product(*zip(box.lower,box.upper))),float)
  radius=float(np.max(np.linalg.norm(corners-np.asarray(workspace.eef_xyz),axis=1)))+.03
  sweep=SimpleNamespace(gripper_radius_m=.035,held_extra_radius_m=max(0.,radius-.035))
 p=proposal_for_p3(name,args,workspace,locate,held_sweep=sweep)
 if held:p=replace(p,operated_entity=b.source)
 # Unheld VLA operator comes from the public language/box goal; never force
 # a door/contact action to be labelled as the source payload.
 return p

def locate(b,prompt,workspace):
 role=infer_public_goal(prompt,source_aliases=b.source_aliases,target_aliases=('microwave','oven'),public_held=workspace.evidence.robot.hold_estimate=='held')
 name=b.source if role=='source'else 'microwave'if role=='target'else None
 box=workspace.box(name)if name else None
 return (name,tuple((lo+hi)/2 for lo,hi in zip(box.lower,box.upper)))if box else None

def workspace(b,entry,sam_client,logs):
 packet=entry.scene.capture_agent();camera=packet['cameras']['agentview'];boxes=[]
 for name,prompt in [('microwave','microwave'),('microwave_door','open microwave door panel'),(b.source,b.source_prompt)]:
  try:
   box,details=sam_public_box(name=name,prompt=prompt,camera=camera,sam3_client=sam_client);boxes.append(box);logs.append({'frame_id':packet['frame_id'],**details})
  except Exception as exc:logs.append({'frame_id':packet['frame_id'],'name':name,'error':f'{type(exc).__name__}: {exc}'})
 return PublicWorkspace(entry.adapter.capture_commit_evidence(),tuple(float(x)for x in packet['robot_state']['eef_pos']),tuple(boxes))
