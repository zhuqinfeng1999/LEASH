"""Private CPU eligibility check for S2 round-2 ketchup transfer.

Only a trusted evaluator-side simulator recorder may supply PrivateSupportSample
objects. The planner's tool result and its text are never proof. This checks
an evidence trace; a native MuJoCo recorder for the basket has not yet been
qualified, so production callers without that recorder must supply no proof.
"""

from dataclasses import dataclass
from math import dist, isfinite

MIN_LIFT_M = .020
MIN_HOLD_S = .50
MIN_SUPPORT_S = .50
MAX_SUPPORT_SPEED_M_S = .010
MAX_SAMPLE_GAP_S = .10
MAX_CENTER_SPEED_M_S = 1.5  # generous CPU guard against pose teleportation
_SEAL = object()


@dataclass(frozen=True)
class PrivateSupportSample:
    sim_time_s: float
    round_index: int
    left_pad_ketchup_contact: bool
    right_pad_ketchup_contact: bool
    ketchup_lift_from_initial_support_m: float
    ketchup_center_xyz_m: tuple[float, float, float]
    gripper_physically_open: bool
    ketchup_center_in_basket_region: bool
    ketchup_basket_contact: bool
    ketchup_nonbasket_contact: bool
    ketchup_linear_speed_m_s: float


class PhysicalSupportProof:
    __slots__ = ("evaluator", "scene", "first_time_s", "last_time_s",
                 "hold_duration_s", "support_duration_s", "_seal")

    def __init__(self, evaluator, scene, first_time_s, last_time_s,
                 hold_duration_s, support_duration_s, seal):
        if seal is not _SEAL:
            raise TypeError("Physical support proof must come from private trace verification")
        self.evaluator = evaluator
        self.scene = scene
        self.first_time_s = first_time_s
        self.last_time_s = last_time_s
        self.hold_duration_s = hold_duration_s
        self.support_duration_s = support_duration_s
        self._seal = seal


def _valid_sample(sample):
    if type(sample) is not PrivateSupportSample or sample.round_index != 2:
        return False
    numeric = (sample.sim_time_s, sample.ketchup_lift_from_initial_support_m,
               sample.ketchup_linear_speed_m_s)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not isfinite(value) for value in numeric):
        return False
    center = sample.ketchup_center_xyz_m
    if (not isinstance(center, (tuple, list)) or len(center) != 3 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not isfinite(v) for v in center)):
        return False
    flags = (sample.left_pad_ketchup_contact, sample.right_pad_ketchup_contact,
             sample.gripper_physically_open, sample.ketchup_center_in_basket_region,
             sample.ketchup_basket_contact, sample.ketchup_nonbasket_contact)
    return all(type(flag) is bool for flag in flags) and sample.ketchup_linear_speed_m_s >= 0


def verify_private_support_trace(evaluator, scene, samples):
    """Return a same-episode proof only after physical grasp, release, support.

    `samples` must be consecutive round-2 controller-step truth samples from a
    trusted recorder. This function rejects missing/malformed/teleport-only
    traces; it cannot attest that an untrusted caller truly read MuJoCo.
    """
    if getattr(evaluator, "_scene", None) is not scene:
        return None
    if not isinstance(samples, (tuple, list)) or len(samples) < 3:
        return None
    if any(not _valid_sample(sample) for sample in samples):
        return None
    for previous, current in zip(samples, samples[1:]):
        dt = current.sim_time_s - previous.sim_time_s
        if dt <= 0 or dt > MAX_SAMPLE_GAP_S:
            return None
        if dist(previous.ketchup_center_xyz_m, current.ketchup_center_xyz_m) > (
                MAX_CENTER_SPEED_M_S * dt + .005):
            return None
    if samples[0].ketchup_center_in_basket_region or samples[0].ketchup_lift_from_initial_support_m > .005:
        return None
    held_since = None
    held_duration = 0.0
    phase = "hold"
    support_since = None
    for sample in samples:
        two_pad_lift = (sample.left_pad_ketchup_contact and
                        sample.right_pad_ketchup_contact and
                        sample.ketchup_lift_from_initial_support_m > MIN_LIFT_M and
                        not sample.gripper_physically_open)
        if phase == "hold":
            held_since = sample.sim_time_s if two_pad_lift and held_since is None else held_since
            if not two_pad_lift:
                held_since = None
            elif sample.sim_time_s - held_since >= MIN_HOLD_S:
                held_duration = sample.sim_time_s - held_since
                phase = "release"
        elif phase == "release":
            if (sample.gripper_physically_open and
                    not sample.left_pad_ketchup_contact and
                    not sample.right_pad_ketchup_contact):
                phase = "support"
        if phase == "support":
            supported = (sample.gripper_physically_open and
                         not sample.left_pad_ketchup_contact and
                         not sample.right_pad_ketchup_contact and
                         sample.ketchup_center_in_basket_region and
                         sample.ketchup_basket_contact and
                         not sample.ketchup_nonbasket_contact and
                         sample.ketchup_linear_speed_m_s <= MAX_SUPPORT_SPEED_M_S)
            support_since = (sample.sim_time_s if supported and support_since is None
                             else support_since if supported else None)
    if phase != "support" or support_since is None:
        return None
    terminal_support_duration = samples[-1].sim_time_s - support_since
    if terminal_support_duration < MIN_SUPPORT_S:
        return None
    return PhysicalSupportProof(
        evaluator, scene, samples[0].sim_time_s, samples[-1].sim_time_s,
        held_duration, terminal_support_duration, _SEAL)


def eligible_proof(proof, evaluator, scene, round_start):
    if type(proof) is not PhysicalSupportProof or proof._seal is not _SEAL:
        return False
    if proof.evaluator is not evaluator or proof.scene is not scene:
        return False
    if 2 not in round_start or 3 not in round_start:
        return False
    return (round_start[2]["sim_time"] <= proof.first_time_s and
            abs(proof.last_time_s - round_start[3]["sim_time"]) <= 1e-6 and
            proof.hold_duration_s >= MIN_HOLD_S and
            proof.support_duration_s >= MIN_SUPPORT_S)
