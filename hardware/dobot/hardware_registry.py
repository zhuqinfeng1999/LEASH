"""Map completed blocking Jodell commands to the original two-event registry.

The simulator HOLD sign may continue a previous OPEN motor target. Our
blocking Jodell command has already completed before a later translation.
Only a fresh public terminal-status receipt can establish that distinction;
occupancy stays UNKNOWN and explicit OPEN still receives its normal review.
"""
from dataclasses import dataclass, replace

from research.v4.leash_v43.registry import Proprio, Match, RELEASE
from research.v4.development.g3_event_count_repair_20261020.registry import CountBoundTransitionRegistry


@dataclass(frozen=True)
class HardwareProprio(Proprio):
    gripper_command_motion_complete: bool | None = None
    gripper_motion_feedback_source: str | None = None


class HardwareCommitRegistry(CountBoundTransitionRegistry):
    def classify(self, cmd, p, context, lease):
        result = super().classify(cmd, p, context, lease)
        completed_hold = (isinstance(p, HardwareProprio)
            and p.gripper_command_motion_complete is True
            and p.gripper_motion_feedback_source == 'CRC-verified Jodell raw status, gOBJ terminal state'
            and cmd.gripper == 'HOLD')
        if completed_hold and result.state == Match.MATCH and RELEASE in result.events:
            events = tuple(event for event in result.events if event != RELEASE)
            return replace(result, state=Match.MATCH if events else Match.NO_MATCH,
                events=events, reason='Actual HOLD does not continue a completed blocking Jodell OPEN'
                if not events else result.reason)
        return result
