"""Own registered I4 cases: current new props with common safe reset pose."""
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.user_rework_support_20260927.i_family import scene as family
from research.v4.candidates.user_rework_support_20260927.i_candle_front_v2 import scene as candle
IDS=('I4-01','I4-02','I4-03')
COMMON_PARK=tuple(tabletop.PARK_QPOS)
BASE_COMMON_CREATE=family.BASE_COMMON_CREATE

def create(instance,seed,**options):
    if instance not in IDS:raise ValueError('Registry has exactly three I4 cases')
    if instance=='I4-01':
        module=family.bases.other.BASES[instance];prior=module.I4HandleLabelEnv
        class CommonParkEnv(prior):
            def __init__(self,*a,**kw):
                with patch.object(tabletop,'PARK_QPOS',COMMON_PARK):super().__init__(*a,**kw)
        with patch.object(module,'I4HandleLabelEnv',CommonParkEnv):return family.create(instance,seed,**options)
    def construct(seed,**kw):
        with patch.object(tabletop,'PARK_QPOS',COMMON_PARK):return BASE_COMMON_CREATE(seed,**kw)
    own=candle.create if instance=='I4-02'else family.create
    with patch.object(family,'BASE_COMMON_CREATE',construct):return own(instance,seed,**options)
