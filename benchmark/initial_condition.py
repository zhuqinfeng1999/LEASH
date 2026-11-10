"""Reconstruct the registered physical initial condition of a simulated trial."""
import argparse
import csv
from pathlib import Path

from benchmark.native import native_scene

BASE = Path(__file__).resolve().parent

def trial_setting(task, planner, method, seed):
    with (BASE / "trial-settings.csv").open(newline="", encoding="utf8") as file:
        matches = [row for row in csv.DictReader(file) if
                   (row["task"], row["planner"], row["method"], int(row["seed"])) == (task, planner, method, seed)]
    if len(matches) != 1:
        raise ValueError(f"Expected one registered trial for {(task, planner, method, seed)}")
    return matches[0]

def apply_setting(scene, row, output):
    if row["initialization"] == "standard":
        return None
    if row["initialization"] != "preheld_tabletop":
        raise ValueError(f"Unrecognized initialization {row['initialization']}")
    from research.v4.development.user_review_revision_20261026 import initialization, source_binding_20261101
    instance = row["task"].replace("T", "R", 1) if row["task"].startswith("T") else row["task"]
    if row["source_binding"] and row["source_binding"] != source_binding_20261101.VERSION:
        raise ValueError("Unknown source-object binding")
    initializer = source_binding_20261101.initialize if row["source_binding"] else initialization.initialize
    receipt = initializer(scene, output=output, instance=instance, seed=int(row["seed"]))
    if instance.startswith("R"):
        from research.v4.development.user_review_revision_20261026.preheld_private_control_20261101 import associate_private_final_control
        associate_private_final_control(scene, receipt)
    invalidate = getattr(scene, "invalidate_initial_occupancy", None)
    if callable(invalidate):
        invalidate()
    if getattr(scene, "public_initial_occupancy", lambda: None)() is not None:
        raise RuntimeError("Pre-held scene still advertises empty occupancy")
    return receipt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--planner", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", default=0, type=int, choices=(0, 1, 2))
    parser.add_argument("--output", type=Path, required=True, help="Fresh path for initialization evidence")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Provide a fresh output path.")
    row = trial_setting(args.task, args.planner, args.method, args.seed)
    with native_scene(args.task, args.seed, "B0" if args.method == "Baseline" else "Ours", camera_observations=True) as scene:
        apply_setting(scene, row, args.output)
        print(f"{args.task}, seed {args.seed}, {row['initialization']}, {row['protocol']}")
