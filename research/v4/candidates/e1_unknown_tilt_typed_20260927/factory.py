from functools import lru_cache
from types import SimpleNamespace
from research.v4.candidates.e1_container_unknown_tilt_dispatch_20260927.factory import components as parent
from .outlet import TypedContainerAdapter
IDS=('E1-05','E1-06')
@lru_cache(maxsize=2)
def components(instance):
    return SimpleNamespace(**{**vars(parent(instance)), 'E1GatedAdapter':TypedContainerAdapter})
