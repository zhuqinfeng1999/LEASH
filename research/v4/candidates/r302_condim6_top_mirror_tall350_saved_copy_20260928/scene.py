"""Exact qualified device delta at own original precompile, no startup steps."""
from pathlib import Path
from types import FunctionType

from research.v4.candidates.r302_source_pair_condim6_20260928 import scene as own
from research.v4.candidates.r305_tall_microwave_inside12_20260928.scene import qualified_device_delta

H = Path(__file__).resolve().parent


class Tall350SourcePairEnv(own.SixDimSourcePairEnv):
    def _load_model(self):
        super()._load_model()
        self.r302_tall350_precompile_receipt = qualified_device_delta(self.model.root)


_RAISED = FunctionType(own._RAISED.__code__, dict(own._RAISED.__globals__, H=H),
                      closure=own._RAISED.__closure__)
_FRONT = FunctionType(own._FRONT.__code__, dict(own._FRONT.__globals__, H=H,
                      _RAISED_INIT=_RAISED, FrontShortVEnv=Tall350SourcePairEnv),
                      closure=own._FRONT.__closure__)


class CompileOnlyTall350CanScene(own.SixDimSourceSupportCanScene):
    """Own original constructor before the fixed80 wrapper, compile-only."""
    __init__ = _FRONT
    layout_version = 'R302-own-shortV-pair6-F-raised-exact-qualified350mm-device-compile-only-v1'

    def step(self, action):
        raise RuntimeError('Compile-only candidate cannot advance real controls')
