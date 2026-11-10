"""Auxiliary transport ability only; original R hazard/refusal labels untouched."""
from dataclasses import asdict
import numpy as np
from research.v4.accelerated.scoring import PlacementTracker
from research.v4.candidates.r4_02_plug_surface_v2_20260926.offline_evaluator import Evaluator as Original
from .geometry import evidence
class Evaluator(Original):
 def __init__(self):super().__init__();self.transport=PlacementTracker();self.last_geometry=None
 def observe(self,s):
  row=super().observe(s);self.last_geometry=evidence(s);obj=s.backend.objects_dict[s.item_name];sim=s.backend.sim
  self.transport.observe(time_s=float(sim.data.time),object_z_m=float(sim.data.get_joint_qpos(obj.joints[0])[2]),attached=bool(s.grasp_assist.held),in_target=bool(self.last_geometry['in_target']),supported=bool(self.last_geometry['floor_supported']),speed_m_s=float(np.linalg.norm(sim.data.get_joint_qvel(obj.joints[0])[:3])))
  return row
 def finish(self,*a,**kw):
  r=super().finish(*a,**kw);r['mechanical_transport_completion']=asdict(self.transport.evidence());r['actual_mattress_geometry']=self.last_geometry;return r
