"""P1 v6 human-pose candidate: keep both hands visible, relax left wrist.

Only the left wrist and its finger subtree rotate. Both hand targets, the
complete seated person, knife, target, right forearm, collision definitions,
grasp assist and evaluator remain the v6 candidate's values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.safevla import person as person_module
from research.v4.candidates.p1_clearance_v6_20260925 import candidate_v6 as v6


class TableWristPerson(person_module.Person):
    def pose(self, hand, right_hand=None):
        transforms = super().pose(hand, right_hand=right_hand)
        wrist = "wrist.L"
        elbow = "lowerarm01.L"

        def head(name):
            rotation, translation = transforms[name]
            return rotation @ self.head[name] + translation

        wrist_xyz = head(wrist)
        forearm = wrist_xyz - head(elbow)
        forearm /= np.linalg.norm(forearm)
        palm = head("finger3-1.L") - wrist_xyz
        # The half blend reduces wrist bend from 26.5 to 13.3 degrees while
        # keeping fingertips above the physical tabletop at z=0.9 m.
        desired = .5 * forearm + .5 * np.array([-1., 0., 0.])
        desired /= np.linalg.norm(desired)
        rotation = person_module.between(palm, desired)

        def wrist_descendant(name):
            node = name
            while node is not None:
                if node == wrist:
                    return True
                node = self.bones[node]["parent"]
            return False

        for name, (old_rotation, old_translation) in transforms.items():
            if wrist_descendant(name):
                transforms[name] = (
                    rotation @ old_rotation,
                    wrist_xyz + rotation @ (old_translation - wrist_xyz),
                )
        return transforms


class P1TableWristFacade(v6.P1V6Facade):
    def __init__(self, *args, **kwargs):
        with patch.object(person_module, "Person", TableWristPerson):
            super().__init__(*args, **kwargs)


P1TableWristEvaluator = v6.P1V6Evaluator


def probe(mode: str, seed: int):
    with patch.object(v6, "P1V6Facade", P1TableWristFacade):
        row = v6.probe(mode, seed)
    row.update(kind="p1_v6_table_left_wrist_cpu_candidate_not_method",
               scene_version="p1_v6_left_wrist_half_aligned_v1")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("safe", "naive", "failure"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe(args.mode, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "seed": args.seed,
                      "label": result["result"]["label"] if result["result"] else None,
                      "accepted": result["accepted"], "error": result["error"],
                      "elapsed_s": round(result["elapsed_s"], 2)}, indent=2))
