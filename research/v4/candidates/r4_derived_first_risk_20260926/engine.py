"""Thin R402/405 configuration over existing geometry/control/first-risk code."""
import inspect
from pathlib import Path
from types import FunctionType
from unittest.mock import patch
from research.v4.candidates.r4_crib_doll_raised3mm_20260926 import candidate as pose,geometry as geometry_engine
from research.v4.candidates.r4_02_plug_surface_v2_20260926 import scripts as cable,project_cpu as cable_project
from research.v4.candidates.r4_05_live_rim_grasp_20260926 import scripts as beads
from research.v4.candidates.r4_05_inclined_identity_v2_20260926 import project_cpu as beads_project
from research.v4.candidates.r4_06_first_risk_stop_20260926 import scripts as stop
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
class Configuration:
 def __init__(self,instance):
  if instance not in('R4-02','R4-05'):raise ValueError('Exact approved derived instances only')
  self.instance=instance;self.Scene=pose.scene_factory(instance);self.project=cable_project if instance=='R4-02'else beads_project
  parent=cable if instance=='R4-02'else beads
  source=inspect.getsource(cable.run_case)if instance=='R4-02'else beads.SOURCE
  if source.count(stop.ANCHOR)!=1:raise RuntimeError('Original actual command/contact boundary changed')
  source=source.replace(stop.ANCHOR,stop.CHECK);self.namespace={**(vars(parent)if instance=='R4-02'else beads.namespace),'ActualRiskStop':stop.ActualRiskStop}
  token='T1R4CableScene'if instance=='R4-02'else'T1R4MagneticBoxScene';self.scene_token=token;self.namespace[token]=self.Scene
  exec(compile(source,__file__,'exec'),self.namespace);self.first_stop=self.namespace['run_case']
 def run_case(self,seed,case):
  try:return self.first_stop(seed,case)
  except stop.ActualRiskStop as stopped:return stopped.report
 def geometry(self,seed):
  with patch.object(geometry_engine,'scene_factory',lambda:self.Scene),patch.object(geometry_engine,'screen',self.project.screen):r=geometry_engine.scan(seed)
  r['instance_id']=self.instance;return r
