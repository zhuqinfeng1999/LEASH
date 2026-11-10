from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.r5_05_laundry_pod_t1_20260926 import scene as old
BINDINGS=tuple(replace(x,xml='research/v4/candidates/r5_05_large_card_v2_20260926/product_card.xml')if x.name=='laundry_product_card_1'else x for x in old.BINDINGS)
def create(*args,**kwargs):
 with patch.object(old,'BINDINGS',BINDINGS):return old.create(*args,**kwargs)
