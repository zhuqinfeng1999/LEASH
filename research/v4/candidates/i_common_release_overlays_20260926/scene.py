"""Shared bottle-only overlay retaining each qualified injection carrier."""
from unittest.mock import patch
from research.v4.candidates.i_common_release_ketchup_20260926.scene import replace_bottle_bindings,make_assist
from research.v4.candidates.i2_natural_corridor_20260925 import scene as i2
from research.v4.candidates.i3_natural_corridor_20260925 import scene as i3
from research.v4.candidates.i4_01_upright_label_20260925 import scene as i4
BASES={'I2-01':i2,'I3-01':i3,'I4-01':i4}
PROTOCOL='i-common-original-cc0-ketchup-authored-body-neck-region-v1'
def create(instance,seed,**kwargs):
    old=BASES[instance]
    with patch.object(old,'BINDINGS',replace_bottle_bindings(old.BINDINGS)):
        env,_unused_old_assist=old.create(seed,**kwargs)
    return env,make_assist(env)
