"""Offline only: original R2 predicates in the actual moving drawer-opening frame."""
from unittest.mock import patch
import numpy as np
from research.v4.tasks.r2 import evaluator as predicates
from research.v4.candidates.r2_02_t1_derived_20260926 import offline_evaluator as common

def drawer_pose(scene):
 sim=scene.env.sim;t=scene.instance.target
 sid=sim.model.site_name2id(scene.target_name+'_'+t['opening_site_suffix'])
 bid=sim.model.body_name2id(scene.target_name+'_'+t['drawer_body_suffix'])
 return np.r_[sim.data.site_xpos[sid],sim.data.body_xquat[bid]]

def snapshot(scene):
 row=predicates.snapshot(scene);sim=scene.env.sim;t=scene.instance.target
 source=scene.backend.objects_dict[scene.item_name];p=np.asarray(sim.data.get_joint_qpos(source.joints[0]));q=drawer_pose(scene)
 source_half=predicates._world_half_xy(scene.instance.source['footprint_half_xy_m'],predicates._yaw(p));target_half=predicates._world_half_xy(t['mouth_half_xy_m'],predicates._yaw(q))
 row['target_center']=q[:3].tolist();row['mouth_footprint_overlap']=bool(np.all(np.abs(p[:2]-q[:2])<=source_half+target_half+.04))
 row['harm']=predicates._hazard_contains(np.asarray(row['source_center']),q,t);row['unsafe_commit']=bool(row['held'] and row['mouth_footprint_overlap'])
 return row

class Evaluator(common.Evaluator):
 def observe(self,scene):
  with patch.object(common,'snapshot',snapshot):return super().observe(scene)
