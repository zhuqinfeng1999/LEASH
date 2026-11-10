"""Public fact from a qualified fresh P1 construction, never dynamic truth.

The original constructor creates new MjData, resets robot and all task bodies,
constructs inactive grasp assistance and refreshes the public sensors. No old
Scene, snapshot or previous Owner queue can be supplied to this constructor.
The small qualification file is offline recipe acceptance, not per-step truth.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import uuid
from research.v4.leash_v43.initial_occupancy import VERSION, empty_reset_fact

ROOT = Path(__file__).resolve().parents[4]
QUALIFICATION = ROOT/'research/v4/results/verified_empty_reset_20261005/initialization_qualification.json'


def wrap_configuration(configured, instance):
    if os.environ.get('SAFEVLA_INITIAL_OCCUPANCY', 'unknown') == 'unknown':
        return configured
    if (os.environ['SAFEVLA_INITIAL_OCCUPANCY'] != VERSION
            or instance not in ('P1-01', 'P1-02')
            or os.environ.get('SAFEVLA_P1_LAYOUT') != 'p1_single_right_arm_dev1'):
        raise ValueError('empty-reset recipe limited to qualified development layouts')
    qualification = json.loads(QUALIFICATION.read_text())
    if qualification['protocol'] != VERSION or qualification['qualified_cases'] != ['P1-01:0', 'P1-02:0']:
        raise ValueError('empty-reset recipe qualification incomplete')
    Original = configured.Scene

    class FreshScene(Original):
        def __init__(self, seed=0, *, camera_observations=False, render_gpu_device_id=-1):
            self._public_initial_fact = None
            if int(seed) != 0:
                raise ValueError('development recipe qualified only for seed zero')
            super().__init__(seed, camera_observations=camera_observations,
                             render_gpu_device_id=render_gpu_device_id)
            if self.sequence != 0:
                self.close()
                raise ValueError('initialization unexpectedly executed a command')
            self._public_initial_fact = empty_reset_fact(
                episode=Path(os.environ['SAFEVLA_RUN_OUTPUT']).name,
                reset_id=uuid.uuid4().hex, initial_step=self.sequence,
                fresh=True, recipe_qualified=True)
            from research.v4.leash_v43.visual_holding import enabled, PublicGraspObservations
            self._public_grasp_observations = (PublicGraspObservations(
                Path(os.environ['SAFEVLA_RUN_OUTPUT']).name) if enabled() else None)

        def step(self, command):
            result = super().step(command)
            observations = getattr(self, '_public_grasp_observations', None)
            if observations is not None:
                observations.executed(command, self.sequence)
            return result

        def public_grasp_observations(self):
            return self._public_grasp_observations

        def public_initial_occupancy(self):
            return None if self._public_initial_fact is None else dict(self._public_initial_fact)

        def invalidate_initial_occupancy(self):
            # State import must invalidate before import. The current episode
            # adapters prohibit resets and expose no snapshot restoration tool.
            self._public_initial_fact = None

        def capture_agent(self):
            from research.v4.bridge.t1_lite import public_capture
            packet = public_capture(self)
            fact = self.public_initial_occupancy()
            if fact is not None:
                packet['robot_state']['initial_occupancy'] = fact
            observations = getattr(self, '_public_grasp_observations', None)
            if observations is not None:
                observations.observe(packet,checkpoint=False)
            return packet

    return SimpleNamespace(**{**vars(configured), 'Scene': FreshScene})
