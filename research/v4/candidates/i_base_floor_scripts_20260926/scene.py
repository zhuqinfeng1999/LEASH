"""Same qualified I base scenes, common CC0 bottle and accepted I1 pixels."""
from research.v4.candidates.i1_legible_paper_native_v2_20260926 import scene as i1
from research.v4.candidates.i_common_release_overlays_20260926 import scene as other
from research.v4.candidates.i_common_release_ketchup_20260926.scene import LEFT_HAND,RIGHT_HAND
IDS=('I1-01','I2-01','I3-01','I4-01')
def create(instance,seed,**options):
    if instance not in IDS:raise ValueError(instance)
    return i1.create(seed,**options)if instance=='I1-01'else other.create(instance,seed,**options)
