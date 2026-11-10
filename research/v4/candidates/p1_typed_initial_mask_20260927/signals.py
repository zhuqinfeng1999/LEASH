"""Structured initial public evidence cause; no hazard/SceneID/phrase rules."""
from research.v4.candidates.e1_02_plain_rpent_v2_20260927.signals import initial_missing_has_other_fault
class MissingPublicEvidence(Exception):
 def __init__(self,missing,perception,*,allowed_names):
  if not missing or not isinstance(allowed_names,frozenset)or not all(isinstance(x,str)and x in allowed_names for x in missing):raise ValueError('Invalid public missing-entity cause')
  self.missing=tuple(sorted(set(missing)));self.perception=tuple(perception)
  super().__init__('Valid initial public capture had no mask for: '+', '.join(self.missing))
 def marker(self):
  return dict(kind='typed_initial_public_masks_missing',version=1,boundary='initial_public_preflight',valid_capture=True,missing_names=list(self.missing),perception=list(self.perception))
