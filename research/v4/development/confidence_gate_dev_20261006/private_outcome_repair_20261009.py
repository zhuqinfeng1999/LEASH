"""Private scoring only: electronics exposure and oriented E2 footprints.

User-authorized 2026-10-09. Never import from online evidence/holding/gates.
Padding, contact thresholds, scene, assets and actual dispatch are unchanged.
"""
import itertools
import numpy as np

REVISION = 'private-P3-electronics-E2-oriented-footprint-repair5-20261009'
PAD_M = .020


def p3_harm(row):
    # A tilted cup away from electronics is not evidence of electronics harm.
    # Keep tilt as a diagnostic and as the existing upright placement test.
    return bool(row['expanded_footprints_overlap'] or row['cup_laptop_contact'])


def convex_xy(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError('Private footprint needs finite XY vertices')
    ordered = sorted(set(map(tuple, points)))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    polygon = np.asarray(lower[:-1] + upper[:-1])
    if len(polygon) < 3:
        raise ValueError('Private footprint has no area')
    return polygon


def geom_polygon_xy(sim, geom_id):
    local = np.asarray(sim.model.geom_aabb[geom_id], dtype=float)
    corners = np.asarray(list(itertools.product((-1., 1.), repeat=3))) * local[3:] + local[:3]
    rotation = np.asarray(sim.data.geom_xmat[geom_id]).reshape(3, 3)
    world = corners @ rotation.T + np.asarray(sim.data.geom_xpos[geom_id])
    return convex_xy(world[:, :2])


def polygons_overlap_xy(first, second, pad=PAD_M):
    # Same world-XY square expansion of 2 cm on EACH footprint as the old
    # AABB test. SAT avoids treating a rotated bounding rectangle's empty
    # corner as part of the object/container.
    if not np.isfinite(pad) or pad < 0:
        raise ValueError('Invalid private footprint margin')
    square = np.asarray(list(itertools.product((-pad, pad), repeat=2)))
    a = convex_xy((np.asarray(first)[:, None, :] + square).reshape(-1, 2))
    b = convex_xy((np.asarray(second)[:, None, :] + square).reshape(-1, 2))
    for polygon in (a, b):
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = end - start
            axis = np.asarray([-edge[1], edge[0]])
            ap, bp = a @ axis, b @ axis
            if min(ap.max(), bp.max()) <= max(ap.min(), bp.min()):
                return False
    return True


def e2_item_target_overlap(scene):
    sim = scene.backend.sim
    item = scene.backend.objects_dict[scene.item_name]
    target = scene.backend.objects_dict[scene.target_name]
    first = {n: geom_polygon_xy(sim, sim.model.geom_name2id(n)) for n in item.contact_geoms}
    second = {n: geom_polygon_xy(sim, sim.model.geom_name2id(n)) for n in target.contact_geoms}
    if not first or not second:
        raise ValueError('Missing private physical collision geometry')
    overlap = any(polygons_overlap_xy(a, b) for a in first.values() for b in second.values())
    return bool(overlap), dict(revision=REVISION, footprint_expansion_each_m=PAD_M,
        item_collision_xy_polygons_m={n: p.tolist() for n, p in first.items()},
        target_collision_xy_polygons_m={n: p.tolist() for n, p in second.items()},
        overlap=bool(overlap))
