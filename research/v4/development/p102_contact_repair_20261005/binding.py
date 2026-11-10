"""Explicit development-only physical handle annotation, shared by B0/Ours."""
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

VERSION = 'cleaver-physical-handle-centre-dev1-20261005'
OLD_CENTER = (-.0725, 0., .028)
HANDLE_CENTER = (-.0725, 0., 0.)
AFFECTED_INSTANCES = ('P1-02',)


def corrected_bindings(bindings):
    """Change only the physical object-local site, never a control target."""
    selected = [b for b in bindings if b.name == 'blunt_tool_1']
    if len(selected) != 1 or tuple(selected[0].grasp_local) != OLD_CENTER:
        raise ValueError('Expected original P1-02 physical handle annotation')
    return tuple(replace(b, grasp_local=HANDLE_CENTER)
                 if b.name == 'blunt_tool_1' else b for b in bindings)


@contextmanager
def binding_scope():
    # The actual constructor consumes this tuple during its normal MJCF build.
    # The original file/accepted defaults and all other instance tuples remain.
    from research.v4.candidates.p1_02_cleaver_t2_20260926 import scene
    with patch.object(scene, 'BINDINGS', corrected_bindings(scene.BINDINGS)):
        yield
