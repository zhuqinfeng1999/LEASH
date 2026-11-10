"""Supply the exact public source field consumed by unchanged common geometry."""
from dataclasses import dataclass
from types import SimpleNamespace

from research.v4.candidates.s101_v43_own_public_consumer_20260929 import runtime as parent


@dataclass(frozen=True)
class OwnSourceBinding:
    source: str = "pan"

    def __post_init__(self):
        if self.source != "pan":
            raise ValueError("only the original own S101 public pan source is selected")


def own_components():
    original = parent.own_public.selected_components()
    if "binding" in vars(original):
        raise RuntimeError("reviewed own selector source seam changed")
    # A private per-instance namespace extension, not a mutation of the parent
    # selector, geometry function globals, Scene, or role/shape algorithm.
    return SimpleNamespace(**vars(original), binding=OwnSourceBinding())


class S101OwnPublicConsumer(parent.S101OwnPublicConsumer):
    def __init__(self, *args, components=None, **kwargs):
        selected = own_components() if components is None else components
        if (type(getattr(selected, "binding", None)) is not OwnSourceBinding
                or selected.binding.source != "pan"
                or selected.original.binding.instance != "S1-01"
                or selected.original.binding.noun != selected.binding.source):
            raise ValueError("exact own S101 public geometry source binding required")
        super().__init__(*args, components=selected, **kwargs)

    # resolve, geometry, roles, lifecycle and outlet are inherited unchanged.
