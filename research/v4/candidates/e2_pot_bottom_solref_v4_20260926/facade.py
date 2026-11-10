"""Add flush interior contact patches; physical occupied volume is unchanged."""
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.e2_storage_castiron_dog_20260926 import facade as original
HERE=Path(__file__).resolve().parent
ROOT=original.ROOT
class E2StablePotDogScene(original.E2StorageCastIronDogScene):
 protocol_version='e2-compact-castiron-bottom-solref020-t5-v42-v4'
 def __init__(self,*args,**kwargs):
  with patch.object(original,'HERE',HERE):super().__init__(*args,**kwargs)
