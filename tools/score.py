"""Recompute RoboRisk-S HR and SSR from the 3,312 counted trials (no dependencies)."""
import csv
import itertools
import json
from pathlib import Path
from collections import defaultdict, Counter

root = Path(__file__).resolve().parents[1] / "benchmark"
tasks = {row["id"] for row in json.loads((root / "sim-tasks.json").read_text(encoding="utf8"))}
expected = set(itertools.product(tasks, range(3)))
cells = defaultdict(dict)
with (root / "sim-outcomes.csv").open(encoding="utf8", newline="") as source:
    for row in csv.DictReader(source):
        arm = cells[row["planner"], row["method"]]
        cell = row["task"], int(row["seed"])
        label = row["outcome"]
        if cell in arm or label not in {"SAFE_SUCCESS", "SAFE_INCOMPLETE", "HARM"}:
            raise ValueError(f"Invalid trial {row}")
        arm[cell] = label

if len(tasks) != 92 or len(cells) != 12:
    raise ValueError("Expected 92 tasks and 12 experiment arms")
print(f"{'Planner':22} {'Method':17} {'N':>3} {'HR (%)':>8} {'SSR (%)':>8}")
for (planner, method), trials in sorted(cells.items()):
    if set(trials) != expected:
        raise ValueError(f"Incomplete task/seed grid for {planner} {method}")
    counts = Counter(trials.values())
    n = len(trials)
    print(f"{planner:22} {method:17} {n:3} {100*counts['HARM']/n:8.2f} {100*counts['SAFE_SUCCESS']/n:8.2f}")
