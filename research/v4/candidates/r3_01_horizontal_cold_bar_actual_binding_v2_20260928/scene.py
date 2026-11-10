"""Opt-in normal R301 constructor; no saved-model or live-state installation."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

from research.v4.accelerated import tabletop
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as carrier
from research.v4.candidates.r3_05_oven_only_prefix_20260928.scene import OwnCartEnv
from research.v4.candidates.r305_tall_microwave_inside12_20260928.scene import qualified_device_delta

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
P = H.parent / 'r3_01_horizontal_cold_bar_saved_continue_20260928'
V2 = H.parent / 'r3_01_horizontal_cold_bar_v2_20260928'
PREPARATION = json.loads((P / 'production_preparation.json').read_text())
REGION = tuple(PREPARATION['grasp_local_private_physics_only'])
INITIAL_JOINTS = tuple(PREPARATION['copied_start_joint_qpos_for_versioned_common_prereset_only'])
ORIGINAL_INIT = carrier.R3TabletopFacade.__init__
ORIGINAL_CART_LOAD = OwnCartEnv._load_model
ORIGINAL_SAMPLE = tabletop.sample_slot
ORIGINAL_BINDINGS = carrier._BINDINGS
ORIGINAL_ENV = carrier._R3FixtureEnv
ORIGINAL_PARK = tabletop.PARK_QPOS
BINDINGS = (
    tabletop.ObjectBinding(carrier.ITEM, str(P / 'moka_horizontal_cold_bar.xml'),
                          'L', z_offset=.118, xy_offset=(0., -.072266002685539),
                          grasp_local=REGION),
    tabletop.ObjectBinding(carrier.OVEN, ORIGINAL_BINDINGS[1].xml, 'R', fixed=True,
                          xy_offset=(.33779602123971007, -.24)),
)


def initial_sample(slot, seed, *, object_index=0):
    """Keep the original draw and add only the declared whole-object yaw."""
    xyz, yaw = ORIGINAL_SAMPLE(slot, seed, object_index=object_index)
    if object_index == 0 and slot == 'L':
        yaw += math.pi / 2
    elif object_index == 1 and slot == 'R':
        yaw -= math.pi / 2
    else:
        raise ValueError('Unexpected own R301 slot/index')
    return xyz, yaw


def add_steel_and_wood(root):
    """Consume the already frozen actual geometry before normal compilation."""
    reference = ET.parse(V2 / 'new_model.xml').getroot()
    receipts = []
    for name in ('fixed_mount0_torso_vis', 'fixed_mount0_pedestal_col',
                 'fixed_mount0_pedestal_vis'):
        nodes = root.findall(".//geom[@name='" + name + "']")
        source = reference.findall(".//geom[@name='" + name + "']")
        if len(nodes) != 1 or len(source) != 1:
            raise ValueError('Exact three-steel precompile seam differs: ' + name)
        before = dict(nodes[0].attrib)
        nodes[0].attrib.clear()
        nodes[0].attrib.update(source[0].attrib)
        receipts.append({'name': name, 'before': before, 'after': dict(nodes[0].attrib)})
    solids = reference.findall(".//geom[@name='r301_source_wood_riser_solid']")
    if len(solids) != 1 or root.findall(".//geom[@name='r301_source_wood_riser_solid']"):
        raise ValueError('Exact one real wood support seam differs')
    # The original frozen block is a world-fixed solid at nominal source XY.
    # It is not resampled, moved after reset, or substituted for the metal foot.
    parent = next(b for b in reference.iter('body') if solids[0] in list(b))
    world = root.find('worldbody')
    if world is None:
        raise ValueError('Normal model worldbody is absent')
    if any(e.tag not in ('geom', 'inertial') for e in parent):
        raise ValueError('Frozen wood body unexpectedly has a joint or child')
    # Object export contains only Moka assets. The real wood's already
    # authored material must accompany its body into the same normal compile.
    assets = root.find('asset')
    source_assets = reference.find('asset')
    if assets is None or source_assets is None:
        raise ValueError('Normal precompile assets are absent')
    for geom in parent.findall('geom'):
        name = geom.get('material')
        if not name:
            continue
        selected = source_assets.findall("material[@name='" + name + "']")
        existing = assets.findall("material[@name='" + name + "']")
        if len(selected) != 1 or len(existing) > 1:
            raise ValueError('Exact frozen real-wood material seam differs')
        if existing:
            if existing[0].attrib != selected[0].attrib:
                raise ValueError('Real-wood material collision')
        else:
            assets.append(deepcopy(selected[0]))
    world.append(deepcopy(parent))
    return receipts


class HorizontalBarEnv(OwnCartEnv):
    def _load_model(self):
        ORIGINAL_CART_LOAD(self)
        self.r301_device_delta = qualified_device_delta(self.model.root)
        self.r301_steel_delta = add_steel_and_wood(self.model.root)


@contextmanager
def constructor_scope():
    if (carrier._BINDINGS is not ORIGINAL_BINDINGS or
            carrier._R3FixtureEnv is not ORIGINAL_ENV or
            tabletop.sample_slot is not ORIGINAL_SAMPLE or
            tabletop.PARK_QPOS is not ORIGINAL_PARK or
            carrier.R3TabletopFacade.__init__ is not ORIGINAL_INIT):
        raise ValueError('Original own constructor scope changed')
    with patch.object(carrier, '_BINDINGS', BINDINGS), \
         patch.object(carrier, '_R3FixtureEnv', HorizontalBarEnv), \
         patch.object(tabletop, 'PARK_QPOS', INITIAL_JOINTS), \
         patch.object(tabletop, 'sample_slot', initial_sample):
        yield


class HorizontalColdBarR301Scene(carrier.R3TabletopFacade):
    layout_version = 'R301-horizontal-cold-U-frame-wood50-steel-base061-tall350-v1'
    common_reset_controls = 20

    def __init__(self, *args, root_candidate_approved=False, **kwargs):
        if root_candidate_approved is not True:
            raise ValueError('Root review required for this new physical candidate')
        from research.v4.leash_v43.run_manifest import source_custody_enabled
        if source_custody_enabled():
            from .admission import admit
            admit()
        with constructor_scope():
            ORIGINAL_INIT(self, *args, **kwargs)
        self.common_initialization_trace = []
        # Robot.reset and its OSC references consume init_qpos normally.
        # No MJB/qpos/warmstart/cached-site state is loaded into this episode.
        from .witness import snapshot, assert_initial
        try:
            self.initialization_before = snapshot(self)
            assert_initial(self, before_neutral=True)
            for index in range(self.common_reset_controls):
                self.step(np.asarray([0., 0., 0., 0., 0., 0., -1.], np.float32))
                row = snapshot(self)
                row['open_neutral_index'] = index + 1
                self.common_initialization_trace.append(row)
                assert_initial(self, before_neutral=False)
            self.initialization_after = snapshot(self)
            if self.sequence != 20 or self.grasp_assist.held:
                raise ValueError('Common20 OPEN initialization accounting differs')
        except BaseException:
            self.close()
            raise

    # step/capture/anchor/close remain the exact original inherited callables.
