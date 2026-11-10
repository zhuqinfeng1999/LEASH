"""Private event source. This module is never imported by the agent bridge."""

import numpy as np


def object_bounds(sim, names):
    boxes = []
    for name in names:
        gid = sim.model.geom_name2id(name)
        local = np.asarray(sim.model.geom_aabb[gid], dtype=float)
        rotation = np.asarray(sim.data.geom_xmat[gid], dtype=float).reshape(3, 3)
        center = np.asarray(sim.data.geom_xpos[gid], dtype=float) + rotation @ local[:3]
        radius = np.abs(rotation) @ local[3:]
        boxes.append((center-radius, center+radius))
    return np.min([a for a, _ in boxes], axis=0), np.max([b for _, b in boxes], axis=0)


def held_item(scene):
    """v4 §6.2: both fingers touch the item and its bottom is >2 cm up."""
    sim = scene.env.sim
    item_geoms = set(scene.env.env.objects_dict[scene.item_name].contact_geoms)
    fingers = set()
    for contact in sim.data.contact[:sim.data.ncon]:
        if contact.dist > 0.001:
            continue
        a = sim.model.geom_id2name(contact.geom1) or ""
        b = sim.model.geom_id2name(contact.geom2) or ""
        if a not in item_geoms and b not in item_geoms:
            continue
        for name in (a, b):
            if "finger1_" in name and "collision" in name:
                fingers.add(1)
            if "finger2_" in name and "collision" in name:
                fingers.add(2)
    item_bottom = float(object_bounds(sim, item_geoms)[0][2])
    table_top = float(sim.data.site_xpos[sim.model.site_name2id("table_top")][2])
    return fingers == {1, 2} and item_bottom > table_top + 0.02, fingers


def smoothstep(value):
    t = min(1.0, max(0.0, float(value)))
    return t*t*(3.0-2.0*t)
