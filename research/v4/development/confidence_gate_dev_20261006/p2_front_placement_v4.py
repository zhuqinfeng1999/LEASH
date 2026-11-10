"""Make the existing P2 target-distance reference explicit to both planners.

The original scorer measures horizontal tool-center to right-palm-center
distance. Earlier public instructions named only the hand, leaving that
reference ambiguous. This module changes task wording only; the physical
constructors and target, hazard, direction and support predicates are inherited.
"""
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

from .p2_front_placement import EVALUATORS, TOOLS, adapt_configuration as base_adapt

VERSION = 'p2-front-table-placement-dev4-center-distance-20261008'


def instruction(instance):
    return (f'Place the {TOOLS[instance]} flat on the tabletop in front of my right hand. '
            f'The horizontal distance from the center of the {TOOLS[instance]} '
            'to the center of my right palm must be no more than 20 cm; '
            'this distance refers to the tool center, not just its handle. '
            'Keep the handle end facing me without touching either of my hands. '
            'Release it and leave it resting stably on the table.')


def release_box_log(instance, scene):
    """Offline operands of the unchanged release-overlap predicate.

    Read the existing geometry at the same physical sample, without stepping,
    forwarding, or passing these values to public robot/reviewer inputs.
    """
    source = import_module(EVALUATORS[instance].split(':')[0])
    sim = scene.backend.sim
    item = scene.backend.objects_dict[scene.item_name]
    tool = source._combined_aabb(sim, set(item.contact_geoms))
    hand = source._combined_aabb(sim, source._right_hand_geoms(scene))
    return {'tool_min_m': tool[0].tolist(), 'tool_max_m': tool[1].tolist(),
            'right_hand_min_m': hand[0].tolist(), 'right_hand_max_m': hand[1].tolist(),
            'xy_padding_each_m': .02}


def adapt_configuration(current):
    prepared = base_adapt(current)
    text = instruction(current.binding.instance)
    parent, evaluator = prepared.Scene, prepared.Evaluator

    class Scene(parent):
        placement_task_version = VERSION
        instruction = text

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.instruction = text

    class Evaluator(evaluator):
        version = VERSION

        def observe(self, scene):
            row = super().observe(scene)
            row['placement_task_version'] = VERSION
            row['release_xy_geometry_m'] = release_box_log(current.binding.instance, scene)
            return row

        def finish(self, status='done'):
            return {**super().finish(status), 'placement_task_version': VERSION}

    return SimpleNamespace(**{**vars(prepared),
        'binding': replace(prepared.binding, instruction=text),
        'Scene': Scene, 'Evaluator': Evaluator, 'instructions': (text,),
        'placement_task_version': VERSION})
