"""Unexecuted own precompile binding; default and original V source untouched."""
from types import FunctionType
from pathlib import Path
from research.v4.candidates.r302_condim6_top_mirror_tall350_saved_copy_20260928 import scene as parent
from research.v4.candidates.r302_front_slot_short_v_20260928 import scene as front
from research.v4.candidates.r302_short_v_raised_neutral_20260928 import scene as raised
from research.v4.candidates.r302_front_slot_fixed80_reset_20260928 import scene as fixed
from .delta import apply


class FlatLedgeTall350Env(parent.Tall350SourcePairEnv):
    def _load_model(self):
        super()._load_model()
        self.r302_flat_ledge_precompile_receipt=apply(self.model.root)


def selected_factory(output,admission):
    """Root-only future constructor, original80 OPEN and existing rest check.

    All old initializer output globals are redirected to one fresh folder.
    This factory is prepared but not instantiated in this source task.
    """
    output=Path(output)
    raise_init=FunctionType(raised.RaisedNeutralShortVCanScene.__init__.__code__,
        dict(raised.RaisedNeutralShortVCanScene.__init__.__globals__,H=output),
        closure=raised.RaisedNeutralShortVCanScene.__init__.__closure__)
    original=front.FrontSlotShortVCanScene.__init__
    front_init=FunctionType(original.__code__,dict(original.__globals__,H=output,
        _RAISED_INIT=raise_init,FrontShortVEnv=FlatLedgeTall350Env),closure=original.__closure__)
    observed=FunctionType(fixed.sample.__code__,dict(fixed.sample.__globals__,H=output),
                         closure=fixed.sample.__closure__)
    initialize=FunctionType(fixed.Fixed80FrontSlotCanScene.__init__.__code__,
        dict(fixed.Fixed80FrontSlotCanScene.__init__.__globals__,H=output,
            _FRONT_INIT=front_init,sample=observed,verify=admission),
        closure=fixed.Fixed80FrontSlotCanScene.__init__.__closure__)
    class FlatLedgeFixed80CanScene(fixed.Fixed80FrontSlotCanScene):
        __init__=initialize
        layout_version='R302-own-flat-gravity-ledge-open-fences-tall350-original400g-v1'
    return FlatLedgeFixed80CanScene
