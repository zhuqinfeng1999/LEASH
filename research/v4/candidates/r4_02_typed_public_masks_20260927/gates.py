"""Keep original ActionGate DENY; attach an exact structured exception witness."""
from dataclasses import dataclass
from research.v4.candidates.r4_02_proceed_outlet_20260927.gates import Toolkit as PreviousToolkit
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PreGateLog
from .public import MissingPublicMasks,valid_marker
@dataclass(frozen=True)
class TypedLog(PreGateLog):
 failure_kind:str|None=None
 public_missing_marker:dict|None=None
class Toolkit(PreviousToolkit):
 def __init__(self,*args,workspace,**kw):
  self._mask_missing=None
  def tracked():
   try:return workspace()
   except MissingPublicMasks as e:self._mask_missing=e.marker();raise
  super().__init__(*args,workspace=tracked,**kw)
 def execute_tool(self,name,args):
  self._mask_missing=None;start=len(self.gate.logs)
  result=super().execute_tool(name,args)
  marker=self._mask_missing
  if marker is not None:
   rows=self.gate.logs[start:]
   if len(rows)!=1:raise RuntimeError('Typed R402 deny-log anchor drift')
   row=rows[0]
   if row.gate!=2 or row.boundary!='planner_to_executor' or row.reviewer_called or not row.system_failure or not valid_marker(marker):raise RuntimeError('Typed R402 evidence/log mismatch')
   self.gate.logs[start]=TypedLog(**vars(row),failure_kind='public_capability_masks_missing',public_missing_marker=marker)
  return result
