"""One missing legacy backend property; original current T1 code retained."""
from pathlib import Path
from research.v4.candidates.r201_current_t1_method_binding_20260928 import binding as parent
from . import sources

HERE = Path(__file__).resolve().parent
VERSION = parent.VERSION + '-legacy-backend-property-v2'


class CurrentR201Scene(parent.CurrentR201Scene):
    @property
    def backend(self):
        return self.env.env


def build_entry(*, version=VERSION):
    if version != VERSION:
        raise ValueError('Unknown current R201 property version')
    pins = sources.admit()
    run = parent.build_entry()
    if run.__globals__['Scene'] is not parent.CurrentR201Scene:
        raise RuntimeError('Original selected T1 Scene seam changed')
    run.__globals__.update(Scene=CurrentR201Scene, hashes=sources.admit, PROTOCOL=VERSION)
    run.native_method_ready = False
    run.launch_authorized = False
    return run
