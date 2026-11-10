"""Private P1 evaluator evidence, not an online holding or grasp mechanism.

Physical finger contact augments the existing assisted grip for both placement
and whole-object forearm exposure. Nothing here creates a weld, changes a
command, or exposes these facts to the public observation pipeline.
"""
from math import isfinite
from types import SimpleNamespace
import re

REVISION = 'p1-private-actual-grip-and-route-coverage-repair8-20261010'
FINGER = re.compile(r'gripper0_right_finger([12])(?:_pad)?_collision')


def contact_evidence(sim, item_names):
    """Use actual item/finger contacts; log operands, not inferred success."""
    own = set(item_names)
    sides, rows = set(), []
    robot_touch = False
    for c in sim.data.contact[:sim.data.ncon]:
        if c.dist > .002:
            continue
        names = [sim.model.geom_id2name(int(c.geom1)) or '',
                 sim.model.geom_id2name(int(c.geom2)) or '']
        if not own.intersection(names):
            continue
        other = names[1] if names[0] in own else names[0]
        robot_touch |= other.startswith(('robot0_', 'gripper0_'))
        match = FINGER.fullmatch(other)
        if match:
            sides.add(int(match.group(1)))
        rows.append(dict(geom1=names[0], geom2=names[1],
                         distance_m=float(c.dist), finger=None if not match else int(match.group(1))))
    # This is the same actual collision-finger channel the original v4 P1
    # task evaluator used; it is not an added pad-force qualification rule.
    return sides, bool(robot_touch), rows


class PhysicalGripCycle:
    """Retain an established physical grasp through one-finger contact gaps.

    A one-sided touch alone cannot establish a grasp. Complete loss of robot
    contact ends the current cycle. A physical drop without actual OPEN cannot
    certify task completion even when it lands inside the receiver.
    """
    def __init__(self):
        self.physical_active = False
        self.cycle_had_physical = False
        self.cycle_open = False
        self.open_sequence = None
        self.ever_physical = False
        self._was_gripped = False
        self._time = None

    def observe(self, *, time_s, assisted, sides, robot_contact, executed_command=None, sequence=None):
        if type(assisted) is not bool or type(robot_contact) is not bool:
            raise ValueError('actual grip flags must be boolean')
        if not isinstance(time_s, (int, float)) or not isfinite(time_s):
            raise ValueError('actual grip time must be finite')
        if self._time is not None and time_s <= self._time:
            raise ValueError('actual grip samples must advance')
        self._time = time_s
        dual = set(sides) == {1, 2}
        if dual:
            self.physical_active = True
        elif not robot_contact:
            self.physical_active = False
        gripped = bool(assisted or self.physical_active)
        if gripped and not self._was_gripped:
            self.cycle_had_physical = False
            self.cycle_open = False
            self.open_sequence = None
        if self.physical_active:
            self.cycle_had_physical = self.ever_physical = True
        if (self._was_gripped or gripped) and executed_command is not None:
            if not isinstance(executed_command, (int, float)) or not isfinite(executed_command):
                raise ValueError('actual executed gripper must be finite')
            if executed_command < 0 and not self.cycle_open:
                self.cycle_open = True
                self.open_sequence = sequence
        self._was_gripped = gripped
        return gripped

    @property
    def release_evidenced(self):
        # Pure assisted-only episodes preserve the original attachment/release
        # contract. Newly recognized physical cycles require the actual OPEN.
        return not self.cycle_had_physical or self.cycle_open


def actual_command(scene):
    row = getattr(scene, 'private_last_executed_gripper', None)
    if not isinstance(row, dict) or row.get('sequence') != scene.sequence:
        return None
    command = row.get('command')
    return command if isinstance(command, (int, float)) and isfinite(command) else None


class ExecutedGripperSceneMixin:
    def __init__(self, *args, **kwargs):
        self.private_last_executed_gripper = None
        super().__init__(*args, **kwargs)

    def step(self, action):
        self.private_last_executed_gripper = None
        result = super().step(action)
        self.private_last_executed_gripper = dict(sequence=self.sequence, command=float(action[6]))
        return result


def configure(configured):
    if not configured.binding.instance.startswith('P1-'):
        return configured
    parent = configured.Scene
    class Scene(ExecutedGripperSceneMixin, parent):
        pass
    return SimpleNamespace(**{**vars(configured), 'Scene': Scene,
                            'placement_outcome_revision': REVISION})
