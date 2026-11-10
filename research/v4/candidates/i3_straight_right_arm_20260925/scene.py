"""I3-01 tablet-screen overlay on the natural-right-arm I1 common tabletop."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.i1_v42_t1t7_20260925 import candidate as i1
from research.v4.candidates.i1_straight_right_arm_20260925.scene import I1StraightRightEnv, BINDINGS
from research.v4.candidates.i3_v42_tablet_20260925.screen import SCREEN_TEXT, OUTPUT

INSTRUCTION = i1.INSTRUCTION
HERE = Path(__file__).resolve().parent


class _I3Env(I1StraightRightEnv):
    @staticmethod
    def person_builder(arena):
        # Keep the accepted I1 scene and seated person's natural right arm.
        # Change only the lower-trust note carrier to a tablet-shaped display.
        I1StraightRightEnv.person_builder(arena)
        if not OUTPUT.is_file():
            raise FileNotFoundError('Generate the source-pinned I3 tablet raster first')
        texture = arena.root.find(".//texture[@name='v4_i1_print']")
        card = arena.root.find(".//body[@name='v4_i1_paper_note']")
        stand = arena.root.find(".//body[@name='v4_i1_note_stand']")
        if texture is None or card is None or stand is None:
            raise RuntimeError('Inherited I1 lower-trust carrier is missing')
        texture.set('file', str(OUTPUT))
        card.set('name', 'v4_i3_tablet')
        substrate = card.find("geom[@name='v4_i1_note_paper']")
        substrate.set('name', 'v4_i3_tablet_substrate')
        substrate.set('rgba', '.055 .078 .105 1')
        # The existing note collision, print mesh, pose and support stay in the
        # same place. A thin, non-colliding dark bezel makes it read as a tablet.
        ET.SubElement(card, 'geom', name='v4_i3_tablet_bezel', type='box',
                      pos='0 0 -.001', size='.221 .131 .002',
                      rgba='.025 .035 .050 1', contype='0', conaffinity='0', group='1')
        ET.SubElement(card, 'geom', name='v4_i3_tablet_camera', type='sphere',
                      pos='0 .124 .002', size='.0028',
                      rgba='.08 .13 .18 1', contype='0', conaffinity='0', group='1')
        post = stand.find("geom[@name='v4_i1_note_post']")
        foot = stand.find("geom[@name='v4_i1_note_foot']")
        post.set('rgba', '.12 .15 .19 1')
        foot.set('rgba', '.12 .15 .19 1')
        arena.tree = ET.ElementTree(arena.root)
        arena.worldbody = arena.root.find('worldbody')
        arena.asset = arena.root.find('asset')


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    """Return the common I1 physics scene with a project-authored tablet UI."""
    key = 'E1-01'
    original = tabletop.PILOTS.get(key)
    tabletop.PILOTS[key] = BINDINGS
    try:
        scene = _I3Env(key, seed, camera_observations=camera_observations,
                       render_gpu_device_id=render_gpu_device_id)
        scene.reset()
        assist = GraspAssist(scene.sim.model._model, scene.sim.data._data,
                             **scene.grasp_assist_kwargs)
        return scene, assist
    finally:
        if original is None:
            del tabletop.PILOTS[key]
        else:
            tabletop.PILOTS[key] = original
