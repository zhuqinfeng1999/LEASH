"""Offline comparison of saved real public proprioception and private QA truth.

This module neither imports the Harness nor collects data. Public and offline
JSONL inputs must come from the same actual executed control, joined by exact
run_id / executed_sequence / sim_time_s. No nearest-frame matching, masks,
default grasp rotation, online access to truth, or scene execution is provided.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite recorded number required")
    return float(value)


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotation(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in value):
        raise ValueError("recorded local-to-world 3x3 rotation required")
    result = [[finite(item) for item in row] for row in value]
    gram = multiply(transpose(result), result)
    determinant = sum(result[0][i] * (result[1][(i + 1) % 3] * result[2][(i + 2) % 3]
                       - result[1][(i + 2) % 3] * result[2][(i + 1) % 3]) for i in range(3))
    if any(abs(gram[i][j] - (1.0 if i == j else 0.0)) > 1e-6 for i in range(3) for j in range(3)) or abs(determinant - 1.0) > 1e-6:
        raise ValueError("recorded orientation is not SO(3)")
    return result


def quaternion_rotation(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("recorded world EEF body quaternion xyzw required")
    q = [finite(item) for item in value]
    length = math.sqrt(sum(item * item for item in q))
    if not 0.9 < length < 1.1:
        raise ValueError("public quaternion norm outside actual adapter domain")
    x, y, z, w = [item / length for item in q]
    return [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]]


def degrees_acos(value):
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def estimated_tilt(current, baseline):
    if current is None or baseline is None:
        return None
    # Original public-body convention: R_current R_grasp.T world_vertical.
    return degrees_acos(multiply(current, transpose(baseline))[2][2])


def orientation_difference(a, b):
    if a is None or b is None:
        return None
    relative = multiply(a, transpose(b))
    return degrees_acos((sum(relative[i][i] for i in range(3)) - 1.0) / 2.0)


def key(row):
    run = row["run_id"]
    seq = row["executed_sequence"]
    if not isinstance(run, str) or not run or type(seq) is not int or seq < 0:
        raise ValueError("recorded run_id and nonnegative integer sequence required")
    return run, seq, finite(row["sim_time_s"])


def read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: object record required")
                rows.append(row)
    return rows


def stats(values):
    if not values:
        return {"n": 0, "mean_absolute_deg": None, "rmse_deg": None, "max_absolute_deg": None}
    return {"n": len(values), "mean_absolute_deg": sum(abs(x) for x in values) / len(values),
            "rmse_deg": math.sqrt(sum(x*x for x in values) / len(values)),
            "max_absolute_deg": max(abs(x) for x in values)}


def onset_summary(points):
    """Summarize recorded angle ranges and adjacent executed 15-degree samples."""
    runs = {}
    for point in points:
        runs.setdefault(point["run_id"], []).append(point)
    summaries = {}
    for run, samples in runs.items():
        summaries[run] = {}
        for mode in ("true_container_tilt_deg", "recorded_R_grasp"):
            rows = []
            for point in samples:
                angle = (point.get(mode) if mode == "true_container_tilt_deg" else
                         point["public_tilt_estimates_deg"][mode])
                # Before actual lift the container resting on the table is
                # not a sample of an estimated carried-container pose.
                lift = point.get("container_lift_m")
                if angle is not None and lift is not None and lift >= .02:
                    rows.append({"executed_sequence": point["executed_sequence"],
                                 "sim_time_s": point["sim_time_s"], "tilt_deg": angle})
            crossings = [{"before": before, "after": after}
                         for before, after in zip(rows, rows[1:])
                         if after["executed_sequence"] == before["executed_sequence"] + 1
                         and before["tilt_deg"] < 15. <= after["tilt_deg"]]
            angles = [row["tilt_deg"] for row in rows]
            summaries[run][mode] = {
                "lifted_2cm_samples": len(rows),
                "minimum_deg": min(angles) if angles else None,
                "maximum_deg": max(angles) if angles else None,
                "below_15deg_samples": sum(angle < 15. for angle in angles),
                "at_or_above_15deg_samples": sum(angle >= 15. for angle in angles),
                "first_at_or_above_15deg": next((row for row in rows if row["tilt_deg"] >= 15.), None),
                "upward_crossings": crossings,
            }
    return summaries


def analyze(public_rows, private_rows, near_onset_halfwidth_deg=2.0):
    private = {}
    for row in private_rows:
        identity = key(row)
        if identity in private:
            raise ValueError("duplicate offline executed frame")
        private[identity] = row
    counts = {"public_records": len(public_rows), "executed_records": 0,
              "withheld_records_ignored": 0, "matched_truth": 0, "missing_truth": 0,
              "holding_UNKNOWN": 0, "holding_HELD": 0, "holding_EMPTY": 0,
              "known_holding_comparisons": 0, "known_holding_matches_weld": 0,
              "false_HELD_vs_weld": 0, "false_EMPTY_vs_weld": 0,
              "weld_latch_comparisons": 0, "weld_latch_disagreements": 0}
    buckets = {mode: {scope: [] for scope in ("all", "lifted_2cm", "near_15deg")}
               for mode in ("recorded_R_grasp", "first_executed_CLOSE_before", "HELD_transition_CLOSE_after")}
    states, points, anchors, gaps, seen = {}, [], [], [], set()
    for public in public_rows:
        if public.get("candidate_dispatched") is not True:
            counts["withheld_records_ignored"] += 1
            continue
        identity = key(public)
        if identity in seen:
            raise ValueError("duplicate public executed frame")
        seen.add(identity)
        run, sequence, sim_time = identity
        state = states.setdefault(run, {"last_sequence": None, "last_gripper": None,
            "last_holding": None, "close_reference": None, "held_reference": None})
        if state["last_sequence"] is not None and sequence != state["last_sequence"] + 1:
            raise ValueError("executed sequence gap: complete command history is required")
        if state["last_sequence"] is None and sequence != 1:
            gaps.append({"run_id": run, "kind": "history_starts_after_sequence_1",
                         "sequence": sequence})
        command = public["command"]
        if not isinstance(command, list) or len(command) != 7:
            raise ValueError("original executed whole 7-vector required")
        command = [finite(item) for item in command]
        gripper = "OPEN" if command[6] < 0 else "CLOSE" if command[6] > 0 else "HOLD"
        after = public["after_public"]
        before = public.get("before_public", {})
        if not isinstance(after, dict) or not isinstance(before, dict):
            raise ValueError("recorded public before/after objects required")
        current = quaternion_rotation(after.get("eef_quat_xyzw"))
        before_rotation = quaternion_rotation(before.get("eef_quat_xyzw"))
        holding = public["holding_estimate"]
        if holding not in {"HELD", "EMPTY", "UNKNOWN"}:
            raise ValueError("actual three-state holding estimate required")
        counts["executed_records"] += 1
        counts["holding_" + holding] += 1
        recorded_baseline = rotation(public.get("R_grasp"))
        if gripper == "OPEN":
            state["close_reference"] = state["held_reference"] = None
        elif gripper == "CLOSE" and state["last_gripper"] != "CLOSE":
            state["close_reference"] = before_rotation
            state["held_reference"] = None
            anchors.append({"run_id": run, "sequence": sequence, "kind": "first_executed_CLOSE_before",
                            "public_R": before_rotation, "arm_rotation_commanded": any(command[3:6]),
                            "qualification": "offline diagnostic only; upright/stable object binding not inferred"})
            if before_rotation is None:
                gaps.append({"run_id": run, "sequence": sequence, "kind": "public_pose_at_CLOSE_missing"})
        if holding == "HELD" and state["last_holding"] != "HELD":
            if gripper == "CLOSE":
                state["held_reference"] = current
            anchors.append({"run_id": run, "sequence": sequence, "kind": "HELD_transition_CLOSE_after",
                            "public_R": state["held_reference"], "recorded_R_grasp": recorded_baseline,
                            "difference_to_first_CLOSE_deg": orientation_difference(current, state["close_reference"])})
        estimates = {"recorded_R_grasp": estimated_tilt(current, recorded_baseline),
                     "first_executed_CLOSE_before": estimated_tilt(current, state["close_reference"]),
                     "HELD_transition_CLOSE_after": estimated_tilt(current, state["held_reference"])}
        truth = private.get(identity)
        point = {"run_id": run, "executed_sequence": sequence, "sim_time_s": sim_time,
                 "holding_estimate": holding, "public_tilt_estimates_deg": estimates,
                 "online_R_grasp_present": recorded_baseline is not None}
        if truth is None:
            counts["missing_truth"] += 1
        else:
            counts["matched_truth"] += 1
            weld, latch = truth.get("weld_active"), truth.get("grasp_latched")
            if type(weld) is bool and holding != "UNKNOWN":
                counts["known_holding_comparisons"] += 1
                counts["known_holding_matches_weld"] += (holding == "HELD") == weld
                counts["false_HELD_vs_weld"] += holding == "HELD" and not weld
                counts["false_EMPTY_vs_weld"] += holding == "EMPTY" and weld
            if type(weld) is bool and type(latch) is bool:
                counts["weld_latch_comparisons"] += 1
                counts["weld_latch_disagreements"] += weld != latch
            container_r = rotation(truth.get("container_R_local_to_world"))
            true_tilt = degrees_acos(container_r[2][2]) if container_r is not None else truth.get("container_tilt_deg")
            true_tilt = None if true_tilt is None else finite(true_tilt)
            lift = truth.get("container_lift_m")
            lift = None if lift is None else finite(lift)
            point.update(weld_active=weld, grasp_latched=latch,
                         true_container_tilt_deg=true_tilt, container_lift_m=lift)
            if true_tilt is not None:
                if not 0 <= true_tilt <= 180:
                    raise ValueError("actual container tilt must be in [0,180] degrees")
                for mode, estimate in estimates.items():
                    if estimate is None:
                        continue
                    error = estimate - true_tilt
                    buckets[mode]["all"].append(error)
                    if lift is not None and lift >= .02:
                        buckets[mode]["lifted_2cm"].append(error)
                    if abs(true_tilt - 15.0) <= near_onset_halfwidth_deg:
                        buckets[mode]["near_15deg"].append(error)
        points.append(point)
        state.update(last_sequence=sequence, last_holding=holding)
        if gripper in {"OPEN", "CLOSE"}:
            state["last_gripper"] = gripper
    total, determinate = counts["executed_records"], counts["known_holding_comparisons"]
    return {"scope": "OFFLINE_REAL_RECORD_DIAGNOSTIC_NOT_MODEL_OR_METHOD_QUALIFICATION",
            "counts": counts, "holding_unknown_fraction": counts["holding_UNKNOWN"] / total if total else None,
            "determinate_holding_weld_agreement": counts["known_holding_matches_weld"] / determinate if determinate else None,
            "near_onset_band_deg": [15.0 - near_onset_halfwidth_deg, 15.0 + near_onset_halfwidth_deg],
            "tilt_error": {mode: {scope: stats(values) for scope, values in scopes.items()}
                           for mode, scopes in buckets.items()},
            "lifted_tilt_ranges_and_15deg_samples": onset_summary(points),
            "anchors": anchors, "gaps": gaps, "points": points,
            "limitations": ["UNKNOWN is neither EMPTY nor a holding classification error.",
                "Private weld/latch/container values are consumed offline only.",
                "First executed CLOSE can miss an object; it is not evidence of grasp or uprightness.",
                "Missing public pose never receives an identity/default rotation.",
                "A measured diagnostic baseline is not supplied to the online registry.",
                "No 15-degree samples or no lift produces n=0, not an accuracy claim."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-jsonl", type=Path, required=True)
    parser.add_argument("--offline-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(read_jsonl(args.public_jsonl), read_jsonl(args.offline_jsonl))
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
