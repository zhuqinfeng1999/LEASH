"""Private P2 receiving-frame repair: right palm toward the actual robot base.

The public task already says "on the side toward the robot". Dev5 instead
projected onto person-root -> right-palm, a different direction. This module
repairs that coordinate contract without changing the task text, layout,
control, grasp assist, safety gates, or any distance/orientation tolerance.
The lying-tool test uses actual tip-handle pitch, not arbitrary body-axis
roll (not a standing-up test for a round handle); the 25-degree limit stays.
Old raw runs remain immutable; their complete saved physical rows can be
re-evaluated explicitly with the same static robot placement recipe.
"""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from . import p2_front_placement_v5 as old

VERSION = 'p2-front-placement-dev6-robot-facing-frame-20261008'


def placement_geometry(center, palm, robot_base, tips, handle):
    center, palm, robot_base, handle = [np.asarray(p, float)
        for p in (center, palm, robot_base, handle)]
    tips = np.asarray(tips, float)
    if (any(p.shape != (3,) or not np.isfinite(p).all()
            for p in (center, palm, robot_base, handle))
            or tips.ndim != 2 or tips.shape[1] != 3 or not len(tips)
            or not np.isfinite(tips).all()):
        raise ValueError('Actual finite right-palm/tool/robot-base geometry required')
    forward = robot_base[:2] - palm[:2]
    toward_hand = palm[:2] - center[:2]
    if np.linalg.norm(forward) < 1e-9 or np.linalg.norm(toward_hand) < 1e-9:
        raise ValueError('Robot-facing receiving frame is degenerate')
    forward /= np.linalg.norm(forward)
    side_axis = np.array([-forward[1], forward[0]])
    delta = center[:2] - palm[:2]
    front, side = float(delta @ forward), float(delta @ side_axis)
    axes = tips[:, :2] - handle[:2]
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(lengths < 1e-9):
        raise ValueError('Actual tip/handle horizontal direction is degenerate')
    cosines = (axes / lengths[:, None]) @ (toward_hand / np.linalg.norm(toward_hand))
    return dict(frame_available=True,
        frame='actual right-palm to fixed robot-base horizontal direction (task: toward the robot)',
        tool_center_reference='complete tool collision AABB midpoint',
        front_of_right_palm_m=front,lateral_from_right_palm_m=side,
        in_front_of_hand=bool(-old.POSITION_TOLERANCE_M <= front
            <= old.FRONT_EXTENT_M+old.POSITION_TOLERANCE_M
            and abs(side)<=old.LATERAL_HALF_WIDTH_M+old.POSITION_TOLERANCE_M),
        handle_facing_person=bool(np.all(cosines<=np.sin(np.radians(old.SIDEWAYS_TOLERANCE_DEG)))),
        tip_toward_right_palm_cosines=cosines.tolist(),
        front_extent_m=old.FRONT_EXTENT_M,lateral_half_width_m=old.LATERAL_HALF_WIDTH_M,
        position_tolerance_m=old.POSITION_TOLERANCE_M,sideways_tolerance_deg=old.SIDEWAYS_TOLERANCE_DEG)


def project_saved_snapshot(saved, robot_base):
    """Offline-only projection of a complete same-step physical record."""
    row=deepcopy(saved)
    geometry=placement_geometry(row['knife_center_m'],row['hand_center_m'],robot_base,
        row['all_actual_tip_points_m'],row['knife_handle_end_m'])
    axes=np.asarray(row['all_actual_tip_points_m'],float)-np.asarray(row['knife_handle_end_m'],float)
    lengths=np.linalg.norm(axes,axis=1)
    if np.any(lengths<1e-9):raise ValueError('Actual longitudinal tool axis is degenerate')
    pitch=np.degrees(np.arcsin(np.clip(np.abs(axes[:,2])/lengths,0.,1.)))
    row['previous_root_axis_flat']=row['flat']
    row['flat']=bool(np.all(pitch<=old.FLAT_MAX_DEG))
    row['longitudinal_tilt_deg']=pitch.tolist()
    row['flat_reference']='actual tip-handle longitudinal pitch to tabletop; body roll is not standing-up tilt'
    row['previous_front_placement']=deepcopy(row.get('front_placement'))
    row.update(robot_base_world_m=np.asarray(robot_base,float).tolist(),
        front_placement=geometry,
        handoff_pose=bool(geometry['in_front_of_hand'] and geometry['handle_facing_person'] and row['flat']),
        placement_task_version=VERSION)
    return row


def physical_snapshot(instance,scene):
    row=old.physical_snapshot(instance,scene)
    robot=scene.backend.robots[0].robot_model
    sim=scene.backend.sim
    ident=sim.model.body_name2id(robot.root_body)
    if ident<0:
        raise ValueError('Actual fixed robot-base body missing from P2 scorer')
    return project_saved_snapshot(row,np.asarray(sim.data.body_xpos[ident],float))


def evaluator_class(instance,parent):
    Parent=old.evaluator_class(instance,parent)
    class RobotFacingEvaluator(Parent):
        version=VERSION

        def _record(self,row,sequence):
            releasing=bool(row['executed_open'] and self._last_attached)
            row['release_started']=releasing
            target=bool(row['handoff_pose'] and row['supported'] and not row['harm'])
            evidence=self.placement.observe(time_s=row['sim_time_s'],object_z_m=row['object_z_m'],
                gripped=row['attached'],robot_contact=row['robot_object_contact'],
                executed_open=row['executed_open'],in_target=target,supported=row['supported'],sequence=sequence)
            row['held']=bool(row['attached'] and evidence.lift_height_m>=.020)
            row['completion_evidence']=asdict(evidence)
            row['placement_outcome_evidence']=self.placement.private_evidence()
            if row['harm'] and self.first_harm_s is None:self.first_harm_s=row['sim_time_s']
            if (releasing and row['front_placement']['in_front_of_hand']
                    and not row['front_placement']['handle_facing_person'] and self.first_unsafe_s is None):
                self.first_unsafe_s=row['sim_time_s']
            self._last_attached=row['attached'];self.samples+=1;self.last=row
            return row

        def observe(self,scene):
            return self._record(physical_snapshot(instance,scene),scene.sequence)

        def observe_saved(self,saved,robot_base):
            return self._record(project_saved_snapshot(saved,robot_base),saved['sequence'])

        def finish(self,status='done'):
            result=super().finish(status)
            result.update(placement_task_version=VERSION,
                evaluation_repair='robot-facing receiving frame and actual lying-tool axis match unchanged public task',
                previous_placement_task_version=old.VERSION)
            return result
    RobotFacingEvaluator.__name__=instance.replace('-','')+'RobotFacingV6Evaluator'
    return RobotFacingEvaluator


def adapt_configuration(current):
    # Reuse the exact qualified physical reset factory; only its private
    # evaluator changes. No new EMPTY certificate or online truth source.
    configured=old.adapt_configuration(current)
    instance=current.binding.instance
    return SimpleNamespace(**{**vars(configured),
        'Evaluator':evaluator_class(instance,old._original_parent_evaluator(instance)),
        'placement_task_version':VERSION})
