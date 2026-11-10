"""Private offline own held-state goal calculation; never exported to planner."""
import itertools
import numpy as np

def derive(scene,details):
 sim=scene.backend.sim;m,d=sim.model._model,sim.data._data
 center=scene.grasp_assist._center;body=m.body(scene.backend.objects_dict[scene.item_name].root_body).id;ov=m.body(scene.backend.objects_dict[scene.oven_name].root_body).id
 E=d.site_xmat[center].reshape(3,3).copy();eef=d.site_xpos[center].copy();bp=d.xpos[body].copy();bR=d.xmat[body].reshape(3,3).copy();op=d.xpos[ov].copy();oR=d.xmat[ov].reshape(3,3).copy()
 hs=m.site('microwave_1_heating_region').id;hp=d.site_xpos[hs].copy();half=m.site_size[hs].copy()
 if not scene.grasp_assist.held:raise RuntimeError('Own actual side55 prefix empty; no tail')
 rel=(E.T@(bp-eef),E.T@bR)
 def bounds(g):
  b=m.geom_aabb[g];G=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g]+G@b[:3]
  return np.array([c+G@(b[3:]*v)for v in itertools.product((-1,1),repeat=3)])
 payload=[g for g in range(m.ngeom)if(m.geom(g).name or '').startswith('steel_thermos_1_')and(m.geom_contype[g]or m.geom_conaffinity[g])]
 hand=[g for g in range(m.ngeom)if(m.geom(g).name or '').startswith('gripper0_')and(m.geom_contype[g]or m.geom_conaffinity[g])]
 if not payload or not hand:raise ValueError('Actual own full tray/closed-hand geometry missing')
 combined=np.concatenate([bounds(g)for g in payload+hand]);offset=(combined-bp)@oR
 def faces(name):return(bounds(m.geom(name).id)-op)@oR
 left=float(faces('microwave_1_g3')[:,0].max());right=float(faces('microwave_1_g4')[:,0].min());floor=float(faces('microwave_1_g7')[:,2].max());roof=float(faces('microwave_1_g6')[:,2].min())
 xi=[left+.005-float(offset[:,0].min()),right-.005-float(offset[:,0].max())];zi=[floor+.005-float(offset[:,2].min()),roof-.005-float(offset[:,2].max())]
 local=oR.T@(bp-op);x=float(np.mean(xi));z=float(np.mean(zi));front=-.1584
 if xi[0]>xi[1]or zi[0]>zi[1]:raise ValueError('Full own tray+closed hand has no actual mouth5mm interval')
 if local[1]+float(offset[:,1].max())>front-.005:raise ValueError('Actual own initial tail station not wholly outside mouth5mm')
 aligned_local=local.copy();aligned_local[0]=x;aligned_body=op+oR@aligned_local;aligned=aligned_body-E@rel[0]
 lowered_local=aligned_local.copy();lowered_local[2]=z;lowered_body=op+oR@lowered_local;lowered=lowered_body-E@rel[0]
 final_local=lowered_local.copy();final_local[1]=float((oR.T@(hp-op))[1]-half[1]+.016);target_body=op+oR@final_local;target=target_body-E@rel[0]
 plan={'high_outside_lateral_alignment':aligned.tolist(),'outside_lower_to_admissible_height':lowered.tolist(),'front_normal_first_original_critical':target.tolist(),'payload_target':target_body.tolist(),'unchanged_EEF_R':E.tolist(),'original_move_tolerance_m':.008,'first_original_critical_margin_m':.016,'tail_stage_caps_proposal':[180,180,300]}
 details['own_actual_held_midpoint_derivation']={'offline_only':True,'actual_sequence':int(scene.sequence),'actual_EEF':eef.tolist(),'actual_payload':bp.tolist(),'actual_payload_R':bR.tolist(),'body_relative_EEF':{'position':rel[0].tolist(),'rotation':rel[1].tolist()},'physical_inner_faces_local':{'left':left,'right':right,'floor':floor,'roof':roof},'body_local_x_interval':xi,'body_local_z_interval':zi,'selected_body_local_x':x,'selected_body_local_z':z,'combined_payload_closed_hand_offsets_low':offset.min(0).tolist(),'combined_payload_closed_hand_offsets_high':offset.max(0).tolist(),'plan':plan,'no_seed0_coordinates_or_qpos_restore':True}
 return plan
