"""Public-data-only E1 descent boundary for a v4.1 action/commit reviewer.

This helper selects *when to ask* for a fresh visual decision; it does not
decide whether the hand is in the target.  No evaluator or scene truth is read.
"""

from __future__ import annotations

import numpy as np


def requires_descent_review(*, held_public: bool, primitive_name: str,
                            current_eef_xyz, proposed_eef_xyz) -> bool:
    """Review a held-object downward motion before sending its first command.

    ``held_public`` is the common Boolean assisted-latch proprioception, not
    private object identity.  The action gate must obtain fresh public RGB-D
    and can deny or constrain the primitive before any descent takes place.
    The commit gate remains responsible for an actual later release.
    """
    if not held_public or primitive_name not in {"move_to", "vla_act"}:
        return False
    current = np.asarray(current_eef_xyz, dtype=float)
    proposed = np.asarray(proposed_eef_xyz, dtype=float)
    if current.shape != (3,) or proposed.shape != (3,):
        raise ValueError("public EEF and proposed target must be 3-vectors")
    if not np.isfinite(current).all() or not np.isfinite(proposed).all():
        raise ValueError("non-finite EEF/target is not an approvable motion")
    return bool(proposed[2] < current[2] - .020)
