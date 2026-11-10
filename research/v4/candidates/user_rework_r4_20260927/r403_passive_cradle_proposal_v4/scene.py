"""Candidate passive stand only, original inflated volume/neck and crib intact."""
from pathlib import Path
from unittest.mock import patch
from ..r403_inflated_scene_v2 import scene as prior
from research.v4.accelerated import tabletop
HERE=Path(__file__).resolve().parent
ROOT=prior.ROOT
ORIGINAL_CONFIGURATION=prior.configuration
def configuration():
 c=ORIGINAL_CONFIGURATION();c['board']['xml']=str(HERE/'passive_stand.xml');return c
class Scene(prior.Scene):
 def __init__(self,*a,**kw):
  sample=tabletop.sample_slot
  def aligned_source(slot,seed,object_index=0):
   return sample(slot,seed,object_index=0 if slot=='L'and object_index==2 else object_index)
  with patch.object(prior,'configuration',configuration),patch.object(tabletop,'sample_slot',aligned_source):super().__init__(*a,**kw)
