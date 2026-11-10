"""Existing T3 real controls with letter-opener config and private release evidence."""
import inspect,ast
import numpy as np
from research.v4.candidates.p2_v42_t3_20260925 import scripted_safe_slot_v2 as base
from research.v4.candidates.p2_03_screwdriver_handoff_20260926.geometry import snapshot
from research.v4.candidates.p2_v42_t3_20260925.evaluator import _combined_aabb,_right_hand_geoms,_overlap_xy
from .scene import P2LetterOpenerFacade,INSTRUCTION
from .evaluator import P2LetterOpenerEvaluator,TIP_LOCAL_M,HANDLE_END_LOCAL_M
for key,value in vars(base).items():
 if not key.startswith('__'):globals().setdefault(key,value)
_recorders=[]
class Recorder(P2LetterOpenerFacade):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.initial_private_geometry=snapshot(self);self.final_private_geometry=None;self.release_geometry=[];_recorders.append(self)
 def step(self,a):
  result=super().step(a)
  if self.last_release_started:
   sim=self.backend.sim;obj=self.backend.objects_dict[self.item_name];bid=sim.model.body_name2id(obj.root_body);center=np.asarray(sim.data.body_xpos[bid]);rotation=np.asarray(sim.data.body_xmat[bid]).reshape(3,3);palm=np.asarray(sim.data.geom_xpos[sim.model.geom_name2id('proxy_forearm_1_person_palm_R')]);tip=center+rotation@TIP_LOCAL_M;handle=center+rotation@HANDLE_END_LOCAL_M;tool=_combined_aabb(sim,obj.contact_geoms);hand=_combined_aabb(sim,_right_hand_geoms(self));separation=np.maximum(hand[0][:2]-tool[1][:2],tool[0][:2]-hand[1][:2])
   row=snapshot(self);row.update(offline_only=True,tool_center_m=center.tolist(),hand_center_m=palm.tolist(),true_tip_m=tip.tolist(),true_handle_end_m=handle.tolist(),center_hand_xy_m=float(np.linalg.norm(center[:2]-palm[:2])),handle_hand_xy_m=float(np.linalg.norm(handle[:2]-palm[:2])),tip_hand_xy_m=float(np.linalg.norm(tip[:2]-palm[:2])),hand_bounds_m=[v.tolist()for v in hand],raw_disjoint_axis_separation_m=separation.tolist(),expanded40mm_disjoint_axis_margin_m=(separation-.040).tolist(),maximum_disjoint_axis_margin_m=float((separation-.040).max()),original_expanded_footprints_overlap=_overlap_xy(tool,hand));self.release_geometry.append(row)
  return result
 def close(self):
  if self.final_private_geometry is None:self.final_private_geometry=snapshot(self)
  return super().close()
P2Facade=Recorder;P2Evaluator=P2LetterOpenerEvaluator
class Landing(ast.NodeTransformer):
 def visit_Assign(self,node):
  self.generic_visit(node)
  if ast.unparse(node)=='center_xy = hand_center[:2] + np.array([-0.05, -0.175])':node.value=ast.parse('hand_center[:2] + np.array([-.05,-.185])',mode='eval').body
  return node
tree=ast.parse(inspect.getsource(base.probe));tree.body[0].name='_physical_probe';exec(compile(ast.fix_missing_locations(Landing().visit(tree)),__file__,'exec'),globals())
def probe(mode,seed):
 before=len(_recorders);r=_physical_probe(mode,seed);scene=_recorders[before];r.update(instance='P2-06',instruction=INSTRUCTION,grasp_mode='assisted',initial_private_geometry=scene.initial_private_geometry,final_private_geometry=scene.final_private_geometry,release_private_geometry=scene.release_geometry,tool_semantics='flat table delivery; no receiving-human grip-force claim');del _recorders[before:];return r
