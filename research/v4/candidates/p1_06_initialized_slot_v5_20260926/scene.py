"""Initialize intended normal bucket slot before common physical weld setup."""
from pathlib import Path
import numpy as np
from research.v4.candidates.p1_06_high_neck_v2_20260926.scene import P1WineBucketFacade as Previous,INSTRUCTION
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
class P1WineBucketFacade(Previous):
 e1_plate_xy_m=(.10,.32)
 protocol_version='p1-06-initialized-nearer-slot-exactshape-v5'
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);self._offline_target_anchor_trace=[];self._audit_anchor()
 def _audit_anchor(self):
  sim=self.backend.sim;m=sim.model._model;d=sim.data;target=self.backend.objects_dict[self.target_name];tid=sim.model.body_name2id(target.root_body);eid=m.equality('e1_plate_fixture').id
  self._offline_target_anchor_trace.append(dict(time_s=float(d.time),actual_target_xyz_m=d.body_xpos[tid].tolist(),fixture_anchor_xyz_m=m.eq_data[eid,3:6].tolist(),fixture_active=bool(d._data.eq_active[eid])))
 def step(self,action):
  r=super().step(action);self._audit_anchor();return r
