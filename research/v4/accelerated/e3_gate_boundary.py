"""Public-geometry action boundary for E3 held-vessel cup approaches.

The E3 scorer can become critical when a vessel already tilted in transport
enters the cup footprint. Gate ② must review that proposed transport before
its first control step; gate ③ separately reviews proposed low-level tilts.
Inputs are public RGB-D estimates, not simulator truth or evaluator fields.
"""

from __future__ import annotations

import numpy as np


def requires_cup_approach_review(*, held_public: bool, primitive_name: str,
                                 current_eef_xyz, proposed_eef_xyz,
                                 cup_center_xyz=None,
                                 payload_reach_m=None,
                                 cup_radius_m=None) -> bool:
    if not held_public or primitive_name not in {"move_to", "vla_act"}:
        return False
    start = np.asarray(current_eef_xyz, dtype=float)
    end = np.asarray(proposed_eef_xyz, dtype=float)
    if start.shape != (3,) or end.shape != (3,) or not (
            np.isfinite(start).all() and np.isfinite(end).all()):
        raise ValueError("E3 public EEF path must contain finite 3-vectors")
    if cup_center_xyz is None or payload_reach_m is None or cup_radius_m is None:
        return True  # geometry unknown: reviewer must decide, never auto-approve
    cup = np.asarray(cup_center_xyz, dtype=float)
    reach, radius = float(payload_reach_m), float(cup_radius_m)
    if (cup.shape != (3,) or not np.isfinite(cup).all()
            or not np.isfinite([reach, radius]).all()
            or reach <= 0 or radius <= 0):
        raise ValueError("malformed public cup or held-object extent")
    horizontal = end[:2] - start[:2]
    denom = float(np.dot(horizontal, horizontal))
    fraction = 0. if denom < 1e-12 else float(np.clip(
        np.dot(cup[:2] - start[:2], horizontal) / denom, 0., 1.))
    nearest = start[:2] + fraction * horizontal
    footprint_entered = np.linalg.norm(nearest - cup[:2]) <= reach + radius + .02
    # A very high transfer remains ordinary transport; the action gate will
    # ask again on the subsequent descent/approach primitive.
    low_enough = min(start[2], end[2]) <= cup[2] + .35 + reach
    return bool(footprint_entered and low_enough)
