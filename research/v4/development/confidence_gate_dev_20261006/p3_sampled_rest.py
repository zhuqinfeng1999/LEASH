"""Private P3 rest measurement at the actual 20 Hz evaluation boundary.

The original instantaneous free-joint velocity is retained as a diagnostic.
Completion uses motion of the same object reference point between consecutive
physical samples, with the existing 10 mm/s and 0.5 s limits. This addresses
contact-solver velocity spikes without counting wall-clock model waits or
inventing missing physical samples. No value enters planner or reviewer input.
"""
from dataclasses import asdict
from math import isfinite, sqrt

from research.v4.accelerated.p3_offline_scoring import P3V42Evaluator
from research.v4.accelerated.scoring import PlacementTracker
from research.v4.tasks.p3.scene import CUP

REVISION = "p3-physical-sample-rest-dev1-20261008"


class SampledRestTracker(PlacementTracker):
    def __init__(self):
        super().__init__()
        self._previous_position = None
        self._previous_position_time = None
        self._position = None
        self.sampled_speed_m_s = None
        self.instantaneous_speed_m_s = None

    def set_position(self, xyz):
        self._position = tuple(float(x) for x in xyz)
        if len(self._position) != 3 or not all(isfinite(x) for x in self._position):
            raise ValueError("P3 completion requires a finite physical object position")

    def observe(self, *, time_s, attached, object_z_m, in_target, supported, speed_m_s):
        self.instantaneous_speed_m_s = None if speed_m_s is None else float(speed_m_s)
        position = self._position
        self._position = None
        if position is None:
            raise ValueError("P3 physical position missing at evaluation boundary")
        dt = None if self._previous_position_time is None else time_s - self._previous_position_time
        # Two missing 20 Hz samples cannot silently become a rest interval.
        timely = dt is not None and 0 < dt <= .050000001
        self.sampled_speed_m_s = (sqrt(sum((x-y)**2 for x,y in
            zip(position, self._previous_position))) / dt if timely else None)
        self._previous_position, self._previous_position_time = position, time_s
        return super().observe(time_s=time_s, attached=attached, object_z_m=object_z_m,
            in_target=in_target, supported=bool(supported and timely),
            # An unknown speed explicitly makes support ineligible above.
            speed_m_s=self.sampled_speed_m_s if timely else
                (self.instantaneous_speed_m_s if self.instantaneous_speed_m_s is not None else 0.0))


class P3SampledRestEvaluator(P3V42Evaluator):
    def __init__(self):
        super().__init__()
        self.placement = SampledRestTracker()

    def observe(self, scene):
        cup = scene.env.env.objects_dict[CUP]
        self.placement.set_position(scene.env.sim.data.get_joint_qpos(cup.joints[0])[:3])
        row = super().observe(scene)
        row.update(rest_measurement_revision=REVISION,
            rest_sampled_speed_m_s=self.placement.sampled_speed_m_s,
            rest_instantaneous_speed_m_s=self.placement.instantaneous_speed_m_s)
        return row

    def finish(self, status=None):
        result = super().finish(status)
        result['rest_measurement_revision'] = REVISION
        result['rest_completion_evidence'] = asdict(self.placement.evidence())
        return result
