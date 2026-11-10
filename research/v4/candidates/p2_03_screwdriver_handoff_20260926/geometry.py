"""Read-only actual arm, public frustum and handle clearance; never online."""
import itertools,mujoco,numpy as np
from research.v4.accelerated.slot_preflight import project
from research.v4.candidates.p2_v42_t3_20260925.native_visual_qa_blade_side_v3 import arm_angles
from research.v4.candidates.p2_v42_t3_20260925.evaluator import _right_hand_geoms,_combined_aabb
from .scene import base

def snapshot(scene):
 sim=scene.backend.sim;m=sim.model;p=scene.person;obj=scene.backend.objects_dict[scene.item_name];protected=set(p.protected);own=set(obj.contact_geoms);contacts=[]
 for c in sim.data.contact[:sim.data.ncon]:
  names=[m.geom_id2name(int(c.geom1))or'',m.geom_id2name(int(c.geom2))or'']
  if c.dist<=.001 and set(names)&protected and(set(names)&own or any(n.startswith(('robot0_','gripper0_'))for n in names)):contacts.append(dict(names=names,distance_m=float(c.dist)))
 _,ids,vertices=p.surface_vertices(np.asarray(base.LEFT_HAND));skin=vertices[ids];over=(np.abs(skin[:,0])<=.45)&(np.abs(skin[:,1])<=.45)
 lo,hi=_combined_aabb(sim,own);corners=[np.asarray(v)for v in itertools.product(*zip(lo,hi))];views={cam:[project(sim,cam,v,margin_px=0)for v in corners]for cam in scene.public_views}
 nearest=None
 handles=[n for n in own if 'handle'in n or'flat_grip_base'in n]
 for h in _right_hand_geoms(scene):
  for b in handles:
   pair=np.empty(6);distance=float(mujoco.mj_geomDistance(m._model,sim.data._data,m.geom_name2id(h),m.geom_name2id(b),10.,pair))
   if nearest is None or distance<nearest['distance_m']:nearest=dict(hand=h,handle=b,distance_m=distance,closest_points_m=pair.reshape(2,3).tolist())
 return dict(time_s=float(sim.data.time),arm_geometry=arm_angles(p),person_risk_contacts=contacts,skin_vertices_below_table_minus2mm=int((over&(skin[:,2]<.898)).sum()),tool_bounds_m=[lo.tolist(),hi.tolist()],public_frusta=views,all_tool_corners_in_both_views=all(v['visible']for a in views.values()for v in a),nearest_finger_handle_geometry=nearest,projection_not_native_occlusion=True)
