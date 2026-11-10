"""The already-repaired common dev2 translation control, without old gates."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import MethodType

from research.v4.leash_v43.adapter import OSCControlSemantics
from research.v4.development.common_translation_dev_20261005.common_motion import (
    UnsupportedMotion, prepare_motion, execute_motion)
from research.v4.development.transport_workspace_dev_20261005.world_reference_controller import (
    VERSION as CONTROL_VERSION, persistent_reference)
from research.v4.development.transport_workspace_dev_20261005.controller_intent import world_reference_intent


class CommonControl:
    """Instance-local controller wrapper; all commands still use env.step."""
    def __init__(self, primitives):
        self.primitives = primitives
        self.adapter = primitives.env
        self.scene = self.adapter.scene
        self.semantics = OSCControlSemantics.from_scene(self.scene)
        self.last_gripper = None
        self.traces = []
        self.restores = []
        self.pending = None

    def executed_gripper(self):
        dispatcher = getattr(self.adapter, 'dispatcher', None)
        if dispatcher is not None:
            # VLA chunk_step also dispatches here, without calling the patched
            # top-level env.step. Use only a completed original send record.
            for row in reversed(dispatcher.logs):
                if row.get('kind') == 'dispatched':
                    return float(row['payload'][6])
            return None
        return self.last_gripper

    @staticmethod
    def parameters(arguments):
        defaults = dict(max_steps=80, gripper=-1., step_clip=.025, tol=.012,
                        action_scale=.05, target_yaw=None, yaw_step_clip=.10)
        return dict(xyz=deepcopy(arguments['xyz']),
                    **{key:deepcopy(arguments.get(key,value)) for key,value in defaults.items()})

    def prepare_candidate(self, name, arguments):
        """Public descriptor and physical execution use one prepared object."""
        self.pending = None
        if name != 'move_to':
            return dict(route='original_primitive', path_known=False,
                        reason='No common-translation descriptor for this primitive')
        parameters = self.parameters(arguments)
        try:
            description = prepare_motion(self.primitives, semantics=self.semantics,
                previous_gripper_command=self.executed_gripper(), public_reader=self.read_pose,
                control_dt_s=float(self.scene.env.env.control_timestep),
                controller_version=CONTROL_VERSION, **parameters)
        except UnsupportedMotion as exc:
            route = dict(route='original_rpent_servo', path_known=False,
                         controller_version=CONTROL_VERSION, reason=str(exc))
            self.pending = (parameters, None, route)
            return deepcopy(route)
        route = dict(route='common_persistent_world_line', path_known=True,
            controller_version=CONTROL_VERSION, world_start=list(description.start.position),
            world_endpoint=list(description.goal), fixed_rotation=[list(r) for r in description.start.rotation],
            reason='Exact prepared dev2 persistent-world line; public start and actual prior gripper command')
        self.pending = (parameters, description, route)
        return deepcopy(route)

    def read_pose(self):
        raw = self.scene.raw
        sequence = getattr(getattr(self.adapter, 'dispatcher', None), 'sequence', self.scene.sequence)
        return dict(robot0_eef_pos=raw['robot0_eef_pos'], robot0_eef_quat=raw['robot0_eef_quat'],
                    sequence=sequence, time_s=float(self.scene.backend.sim.data.time))

    def _replace(self, obj, name, value):
        had = name in obj.__dict__
        old = obj.__dict__.get(name)
        setattr(obj, name, value)
        def restore():
            if had:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self.restores.append(restore)

    def install(self):
        original_step = self.adapter.step
        def step(_, action):
            result = original_step(action)
            # A withheld candidate or failed send never updates this record.
            self.last_gripper = float(action[6])
            return result
        self._replace(self.adapter, 'step', MethodType(step, self.adapter))
        original_move = self.primitives.move_to
        def move_to(_, xyz, *, max_steps=80, gripper=-1., step_clip=.025,
                    tol=.012, action_scale=.05, target_yaw=None, yaw_step_clip=.10):
            parameters = dict(xyz=xyz, max_steps=max_steps, gripper=gripper,
                step_clip=step_clip, tol=tol, action_scale=action_scale,
                target_yaw=target_yaw, yaw_step_clip=yaw_step_clip)
            if self.pending is None:
                self.prepare_candidate('move_to',parameters)
            if self.pending[0] != parameters:
                raise ValueError('Actual move_to parameters differ from the scored execution route')
            _, description, route = self.pending
            self.pending = None
            if description is None:
                # Existing yaw/transitional-gripper cases retain the original
                # primitive, identically for B0 and both gated arms.
                return original_move(**parameters)
            trace = []
            dispatcher = getattr(self.adapter, 'dispatcher', None)
            with persistent_reference(self.scene, description, trace, primitives=self.primitives), \
                    (world_reference_intent(dispatcher, description, trace)
                     if dispatcher is not None else nullcontext()):
                try:
                    return execute_motion(self.primitives, description,
                                          trace=trace, public_reader=self.read_pose)
                finally:
                    self.traces.append(dict(description=description.as_dict(), trace=trace))
        self._replace(self.primitives, 'move_to', MethodType(move_to, self.primitives))
        return self

    def close(self):
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()


@contextmanager
def installation():
    """Install before RPent caches its primitive handlers; restore on exit."""
    from robots.libero.toolkit import LiberoToolkit
    original = LiberoToolkit.init_primitives_clean
    managers = {}
    def init(toolkit, **kwargs):
        original(toolkit, **kwargs)
        manager = CommonControl(toolkit._primitives)
        managers[id(toolkit._primitives)] = manager
        manager.install()
    LiberoToolkit.init_primitives_clean = init
    try:
        yield managers
    finally:
        LiberoToolkit.init_primitives_clean = original
        for manager in managers.values():
            manager.close()
