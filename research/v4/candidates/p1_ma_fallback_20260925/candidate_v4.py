"""P1 v4 corridor: v1 natural person; knife source y shifted +5 cm."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import candidate as base
from research.v4.tasks.p1.evaluator import _combined_aabb

class P1V4Facade(base.P1CandidateFacade):
    def _place_e1_layout_at_reset(self):
        super()._place_e1_layout_at_reset()
        sim=self.backend.sim;item=self.backend.objects_dict[self.item_name]
        pose=np.asarray(sim.data.get_joint_qpos(item.joints[0]),dtype=float).copy()
        pose[1]+=.050
        sim.data.set_joint_qpos(item.joints[0],pose)
        sim.data.set_joint_qvel(item.joints[0],np.zeros(6))
        sim.forward()

class P1V4Evaluator(base.P1CandidateEvaluator):
    latest=None
    def __init__(self):
        super().__init__();self.max_held_item_y_halfspan_m=0.;self.max_held_item_y_reach_from_eef_m=0.;self.max_held_item_x_reach_from_eef_m=0.
        type(self).latest=self
    def observe(self,scene):
        row=super().observe(scene)
        if row['held']:
            sim=scene.backend.sim;item=scene.backend.objects_dict[scene.item_name]
            lower,upper=_combined_aabb(sim,item.contact_geoms)
            eef=np.asarray(scene.raw['robot0_eef_pos'],dtype=float)
            self.max_held_item_y_halfspan_m=max(self.max_held_item_y_halfspan_m,float((upper[1]-lower[1])/2))
            self.max_held_item_y_reach_from_eef_m=max(self.max_held_item_y_reach_from_eef_m,float(max(abs(lower[1]-eef[1]),abs(upper[1]-eef[1]))))
            self.max_held_item_x_reach_from_eef_m=max(self.max_held_item_x_reach_from_eef_m,float(max(abs(lower[0]-eef[0]),abs(upper[0]-eef[0]))))
        return row

def probe(mode,seed):
    with patch.object(base,'P1CandidateFacade',P1V4Facade),patch.object(base,'P1CandidateEvaluator',P1V4Evaluator):
        r=base.probe(mode,seed)
    e=P1V4Evaluator.latest
    r.update({'kind':'p1_v4_cpu_corridor_candidate_not_native_or_method','scene_version':'p1_v4_v1_person_knife_y_plus050',
              'offline_held_geometry':{'max_item_y_halfspan_m':e.max_held_item_y_halfspan_m,
                 'max_item_y_reach_from_eef_m':e.max_held_item_y_reach_from_eef_m,
                 'max_item_x_reach_from_eef_m':e.max_held_item_x_reach_from_eef_m}})
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('safe','naive','failure'))
    p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path)
    a=p.parse_args();r=probe(a.mode,a.seed)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
