"""The frozen physical source Scene and fixed80 initializer; offline QA only."""
from pathlib import Path
from types import FunctionType

from research.v4.candidates.r302_front_pair_guard_v2_20260928 import scene as previous
from .admit import verify
from .support import sample

H = Path(__file__).resolve().parent
_RAISED = FunctionType(previous._RAISED.__code__, dict(previous._RAISED.__globals__, H=H),
                       closure=previous._RAISED.__closure__)
_FRONT = FunctionType(previous._FRONT.__code__, dict(previous._FRONT.__globals__, H=H, _RAISED_INIT=_RAISED),
                     closure=previous._FRONT.__closure__)
_SELECTED = FunctionType(previous.selected_front_init.__code__,
                        dict(previous.selected_front_init.__globals__, H=H, _FRONT=_FRONT),
                        closure=previous.selected_front_init.__closure__)
_INIT = FunctionType(previous._INIT.__code__,
                     dict(previous._INIT.__globals__, H=H, _FRONT_INIT=_SELECTED, sample=sample, verify=verify),
                     closure=previous._INIT.__closure__)


class SourceSupportClassificationV3CanScene(previous.SourcePairGuardV2CanScene):
    __init__ = _INIT
    layout_version = 'R302-same-two-support-pairs-fixed80-offline-compression-classification-v3'
