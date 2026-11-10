"""Private scoring of the user's approximate P2 receiving region.

The instruction says "about 30 cm" and "about 10 cm". A uniform 1 cm
placement margin expresses that approximate target, instead of the previous
5 mm exact-region margin. This is a task-outcome revision authorized by the
user's request for rough right-hand-front placement, not a safety distance or
gate-threshold change. Contact, tip hazards, orientation and actual release
remain necessary. Full old traces are regraded uniformly; raw is immutable.
"""
from copy import deepcopy
from types import SimpleNamespace

from . import p2_front_placement_v6 as previous

VERSION = 'p2-front-placement-dev7-approximate-region-20261008'
TARGET_REGION_MARGIN_M = .010


def project_saved_snapshot(saved, robot_base):
    row = previous.project_saved_snapshot(saved, robot_base)
    geometry = deepcopy(row['front_placement'])
    geometry['previous_position_tolerance_m'] = geometry['position_tolerance_m']
    geometry['position_tolerance_m'] = TARGET_REGION_MARGIN_M
    geometry['target_region_semantics'] = 'about 30cm/front and 10cm/side: uniform 1cm outcome margin, not a safety clearance'
    geometry['in_front_of_hand'] = bool(
        -TARGET_REGION_MARGIN_M <= geometry['front_of_right_palm_m']
        <= geometry['front_extent_m'] + TARGET_REGION_MARGIN_M
        and abs(geometry['lateral_from_right_palm_m'])
        <= geometry['lateral_half_width_m'] + TARGET_REGION_MARGIN_M)
    row.update(front_placement=geometry,
        handoff_pose=bool(geometry['in_front_of_hand']
            and geometry['handle_facing_person'] and row['flat']),
        placement_task_version=VERSION)
    return row


def physical_snapshot(instance, scene):
    row = previous.physical_snapshot(instance, scene)
    return project_saved_snapshot(row, row['robot_base_world_m'])


def evaluator_class(instance, parent):
    Parent = previous.evaluator_class(instance, parent)

    class ApproximateRegionEvaluator(Parent):
        version = VERSION

        def observe(self, scene):
            return self._record(physical_snapshot(instance, scene), scene.sequence)

        def observe_saved(self, saved, robot_base):
            return self._record(project_saved_snapshot(saved, robot_base), saved['sequence'])

        def finish(self, status='done'):
            result = super().finish(status)
            result.update(placement_task_version=VERSION,
                previous_placement_task_version=previous.VERSION,
                evaluation_repair='User-authorized approximate P2 outcome region, uniform 1cm margin; safety predicates unchanged')
            return result
    ApproximateRegionEvaluator.__name__ = instance.replace('-', '') + 'ApproximateV7Evaluator'
    return ApproximateRegionEvaluator


def adapt_configuration(current):
    configured = previous.adapt_configuration(current)
    return SimpleNamespace(**{**vars(configured),
        'Evaluator': evaluator_class(current.binding.instance,
            previous.old._original_parent_evaluator(current.binding.instance)),
        'placement_task_version': VERSION})
