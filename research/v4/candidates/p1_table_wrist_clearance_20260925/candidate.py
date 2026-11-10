"""Isolated P1 pose check: lift the table-side left hand by 25 mm.

This inherits the protected right arm, route, evaluator, and half-aligned
left wrist of the prior table-wrist candidate. It is not a method result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from research.v4.candidates.p1_clearance_v6_20260925 import candidate_v6 as v6
from research.v4.candidates.p1_natural_left_rest_20260925.candidate_table_wrist import P1TableWristFacade


class P1TableWristClearanceFacade(P1TableWristFacade):
    e1_left_rest_xyz_m = (.20, -.38, .963)


def probe(mode: str, seed: int):
    with patch.object(v6, "P1V6Facade", P1TableWristClearanceFacade):
        row = v6.probe(mode, seed)
    row.update(kind="p1_table_wrist_clearance_cpu_candidate_not_method",
               scene_version="p1_v6_table_left_wrist_lift25mm_v1")
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
                      "accepted": result["accepted"], "error": result["error"]}, indent=2))
