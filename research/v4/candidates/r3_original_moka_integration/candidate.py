"""Isolated R3 binding for the project-authored moka candidate.

This module leaves the active R3 binding, LIBERO source and private evaluator
unchanged. Registration is process-local and must precede scene construction.
"""
from dataclasses import replace
from pathlib import Path

from research.v4.tasks.r3 import template

ASSET_DIR = template.ROOT / "research/v4/tasks/r3/original_moka_candidate"
OBJECT_XML = Path(__file__).resolve().parent / "moka_libero_support.xml"
CATEGORY = "original_moka_pot"
OBJECT_NAME = "moka_pot_1"


def register_original_moka():
    from libero.libero.envs.base_object import OBJECTS_DICT, register_object
    from robosuite.models.objects import MujocoXMLObject

    existing = OBJECTS_DICT.get(CATEGORY)
    if existing is not None:
        if getattr(existing, "candidate_xml", None) != str(OBJECT_XML):
            raise RuntimeError(f"LIBERO category collision: {CATEGORY}")
        return existing

    @register_object
    class OriginalMokaPot(MujocoXMLObject):
        candidate_xml = str(OBJECT_XML)

        def __init__(self, name=OBJECT_NAME, joints=None):
            if joints is None:
                joints = [dict(type="free", damping="0.0005")]
            super().__init__(self.candidate_xml, name=name, joints=joints,
                             obj_type="all", duplicate_collision_geoms=False)
            self.category_name = CATEGORY
            self.rotation = (0, 0)
            self.rotation_axis = "x"
            self.object_properties = {"vis_site_names": {}}

    assert OBJECTS_DICT[CATEGORY] is OriginalMokaPot
    return OriginalMokaPot


def candidate_instance():
    base = template.load_instance("R3-01")
    if base.instruction != template.INSTRUCTION or base.metal_item["name"] != OBJECT_NAME:
        raise RuntimeError("Frozen R3 instruction or object name changed")
    item = dict(base.metal_item)
    item.update(category=CATEGORY, name=OBJECT_NAME,
                display_name="摩卡壶（项目原创候选）",
                xml=str(OBJECT_XML.relative_to(template.ROOT)),
                license_status="project-authored CC0 proposed; owner confirmation pending")
    return replace(base, metal_item=item)


def candidate_bddl():
    return template.render_bddl(candidate_instance())
