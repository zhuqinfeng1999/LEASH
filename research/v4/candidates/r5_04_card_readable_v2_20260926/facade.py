"""Only the identification card changes; bottle/controller/policy unchanged."""
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.r5_04_prescription_t1_20260926 import scene as original_scene
from research.v4.candidates.r5_04_prescription_t1_20260926.facade import R5PrescriptionFacade
PREFIX='research/v4/candidates/r5_04_card_readable_v2_20260926/'
BINDINGS=(*original_scene.BINDINGS[:2],replace(original_scene.BINDINGS[2],xml=PREFIX+'product_card.xml'))
class ReadableCardFacade(R5PrescriptionFacade):
 def __init__(self,*args,**kwargs):
  with patch.object(original_scene,'BINDINGS',BINDINGS):super().__init__(*args,**kwargs)
