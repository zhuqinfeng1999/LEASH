"""One T7 configuration overlay over the current common I tabletop."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.i_common_release_ketchup_20260926 import scene as base
from research.v4.candidates.i2_straight_right_arm_20260925.scene import I2StraightRightEnv
from research.v4.candidates.i3_straight_right_arm_20260925.scene import _I3Env
from research.v4.candidates.i4_01_natural_corridor_20260925.scene import I4HandleLabelEnv
from research.v4.candidates.i4_01_upright_label_20260925.scene import INSPECTION_PARK_QPOS
from .config import row,TARGETS,ROOT
HERE=Path(__file__).resolve().parent
LEFT_HAND=base.LEFT_HAND;RIGHT_HAND=base.RIGHT_HAND
def create(instance,seed,**options):
    r=row(instance);folder=HERE/'generated'/instance;image=folder/'text.png'
    if not image.is_file():raise FileNotFoundError('generate exact instance text first')
    family=r['base_task'];original={'I1':base.I1StraightRightEnv,'I2':I2StraightRightEnv,'I3':_I3Env,'I4':I4HandleLabelEnv}[family]
    class ConfigEnv(original):
        @staticmethod
        def person_builder(arena):
            original.person_builder(arena)
            if family!='I4':arena.root.find(".//texture[@name='v4_i1_print']").set('file',str(image))
    bindings=base.BINDINGS
    if family=='I4':
        target=TARGETS[r['slots']['dangerous_target']]
        bindings=tuple(replace(b,xml=str((folder/'tagged_object.xml').relative_to(ROOT)))if b.name==target else b for b in bindings)
    with patch.object(base,'I1StraightRightEnv',ConfigEnv),patch.object(base,'BINDINGS',bindings):
        if family=='I4':
            with patch.object(tabletop,'PARK_QPOS',INSPECTION_PARK_QPOS):return base.create(seed,**options)
        return base.create(seed,**options)
