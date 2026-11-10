"""Controller-linear translation for a hardware move_to, without OSC microsteps.

The planner schema stays unchanged. This is explicitly a hardware execution
adaptation, not a claim to run the simulator's OSC servo at real-time speed.
Every device send still traverses the existing preflight and gated outlet.
"""
import numpy as np

REVISION = "20261026-controller-linear-waypoint-v2"


def enabled(driver):
    return getattr(driver, "config", {}).get("hardware_move_to_mode") == "controller_linear_waypoint"


def execute(native, primitives, arguments):
    target = np.asarray(arguments["xyz"], dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Finite base-frame target XYZ required")
    if arguments.get("target_yaw") is not None:
        raise ValueError("Continuous translation does not implement simultaneous yaw")
    tolerance = arguments.get("tol", 0.012)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive finite target tolerance required")
    current = np.asarray(primitives._last_obs_eef_pos, dtype=float)
    delta = target - current
    command_count = 0
    # A full-target controller move is not an OSC convergence loop. The
    # reporting tolerance must not silently swallow a requested 10 mm descent
    # merely because the simulator default tolerance is 12 mm.
    command_deadband = min(tolerance, .001)
    if np.linalg.norm(delta) >= command_deadband:
        # This routine calls the same preflight before the jaw command. A
        # rejected full path therefore never opens/closes the gripper first.
        packet = native.step_physical(np.r_[delta, np.zeros(3), arguments.get("gripper", -1.)])
        primitives.set_obs(packet["observation"])
        command_count = 1
    final = np.asarray(primitives._last_obs_eef_pos, dtype=float)
    error = float(np.linalg.norm(target - final))
    return dict(name="move_to", target_xyz=target.tolist(), final_eef_pos=final.tolist(),
                final_dist_m=error, target_reached=error < tolerance,
                steps_used=command_count, max_steps=arguments.get("max_steps", 80),
                terminated=False, truncated=False,
                hardware_execution=dict(revision=REVISION, mode="controller_linear_waypoint",
                    command_deadband_m=command_deadband,
                    logical_waypoint_commands=command_count,
                    simulator_interpolation_parameters_not_applied={k: arguments.get(k, v)
                        for k, v in (("step_clip", .025), ("action_scale", .05))}))
