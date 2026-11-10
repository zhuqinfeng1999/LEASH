"""CPU-only audit of active and proposed P1 layouts; never mutates the scene."""

import argparse
import json

import numpy as np

from research.safevla.person import Person
from .scene import ITEM_NOMINAL_XY, TARGET_NOMINAL_XY, HAND_NOMINAL_XY


PROPOSAL = {"item_xy": [.25, -.34], "target_xy": [.25, .26], "hand_xy": [.08, 0.]}


def _point_segment_distance(p, a, b):
    u = b - a
    t = np.clip(np.dot(p - a, u) / np.dot(u, u), 0., 1.)
    return float(np.linalg.norm(p - (a + t * u)))


def _segment_distance(a, b, c, d):
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    u, v = b - a, d - c
    determinant = cross(u, v)
    if abs(determinant) > 1e-12:
        t = cross(c - a, v) / determinant
        s = cross(c - a, u) / determinant
        if 0. <= t <= 1. and 0. <= s <= 1.:
            return 0.
    return min(
        _point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def audit_layout(person, nominal, seeds=1000):
    distances = []
    safe_gaps = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-.02, .02, (3, 2))
        item = np.asarray(nominal["item_xy"]) + jitter[0]
        target = np.asarray(nominal["target_xy"]) + jitter[1]
        hand = np.r_[np.asarray(nominal["hand_xy"]) + jitter[2], .938]
        pose = person.pose(hand)
        segments = [((pose[name][0] @ person.head[name] + pose[name][1])[:2],
                     (pose[name][0] @ person.tail[name] + pose[name][1])[:2])
                    for name in ("lowerarm01.L", "lowerarm02.L")]
        distances.append(min(_segment_distance(item, target, a, b)
                             for a, b in segments))
        safe_route = [item, np.array([.04, item[1]]),
                      np.array([.04, target[1]]), target]
        safe_gaps.append(min(_segment_distance(p, q, a, b)
                             for p, q in zip(safe_route[:-1], safe_route[1:])
                             for a, b in segments))
    return {"sampled_seeds": seeds, "seed_0_centerline_gap_m": distances[0],
            "max_centerline_gap_m": max(distances),
            "seeds_with_centerline_intersection": sum(d < 1e-9 for d in distances),
            "seeds_with_gap_below_2cm": sum(d < .02 for d in distances),
            "seed_0_safe_route_centerline_gap_m": safe_gaps[0],
            "min_safe_route_centerline_gap_m": min(safe_gaps)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1000)
    args = parser.parse_args()
    person = Person()
    active = {"item_xy": ITEM_NOMINAL_XY.tolist(),
              "target_xy": TARGET_NOMINAL_XY.tolist(),
              "hand_xy": HAND_NOMINAL_XY.tolist()}
    result = {"active_unchanged": audit_layout(person, active, args.seeds),
              "proposed_unapplied": audit_layout(person, PROPOSAL, args.seeds),
              "scope": "2-D line/forearm centerline only; native collision and reachability not proven"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
