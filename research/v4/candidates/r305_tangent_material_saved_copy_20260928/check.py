"""One analytic horizontal material frame on the frozen front-source saved model."""
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import textwrap

import numpy as np

from research.v4.candidates.r305_radial_rim_saved_copy_20260928 import check as parent

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
SAVED = H.parent / 'r305_front_source_material_copy_20260928'


def tangent_material_frame(m, d, center, radial):
    # Use the actual compiled hard palm envelope in EEF-local coordinates.
    # The original radial jaw-X and original material midpoint stay fixed.
    hand = m.geom('gripper0_right_hand_collision').id
    if not (m.geom_contype[hand] or m.geom_conaffinity[hand]):
        raise RuntimeError('Exact hard palm collision missing')
    E = d.site_xmat[center].reshape(3, 3)
    ep = d.site_xpos[center].copy()
    G = d.geom_xmat[hand].reshape(3, 3)
    a = m.geom_aabb[hand]
    world = np.asarray([d.geom_xpos[hand] + G @ (a[:3] + a[3:] * np.asarray(s))
                        for s in itertools.product((-1, 1), repeat=3)])
    local = (world - ep) @ E
    lo, hi = local.min(0), local.max(0)
    tangent = np.cross([0., 0., 1.], radial)
    tangent /= np.linalg.norm(tangent)
    if abs(tangent[1]) < 1.e-10 or lo[2] * hi[2] <= 0.:
        raise RuntimeError('Compiled palm does not determine a unique front-side tangent')
    # Negative world Y is the actual source-table front. The palm lies behind
    # the material approach along local Z; choose the unique sign placing its
    # entire Z-axis envelope toward that front, rather than test two poses.
    sign = 1. if ((lo[2] + hi[2]) / 2.) * tangent[1] < 0. else -1.
    approach = sign * tangent
    R = np.column_stack((radial, np.cross(approach, radial), approach))
    if not np.allclose(R.T @ R, np.eye(3), rtol=0., atol=1.e-12) or np.linalg.det(R) < .999999:
        raise RuntimeError('Derived single material frame is not right handed')
    receipt = {'version': 'R305-front-source-positive-rim-horizontal-tangent-v1',
               'hard_palm_geom': m.geom(hand).name,
               'hard_palm_EEF_local_min': lo.tolist(), 'hard_palm_EEF_local_max': hi.tolist(),
               'hard_palm_local_center': ((lo + hi) / 2.).tolist(),
               'jaw_axis_world': radial.tolist(), 'unsigned_tangent_world': tangent.tolist(),
               'selected_tangent_sign': sign, 'approach_axis_world': approach.tolist(),
               'palm_Z_component_front_shift_interval_m': sorted((approach[1] * lo[2], approach[1] * hi[2])),
               'target_R': R.tolist(), 'no_two_sign_or_angle_trial': True,
               'source_layout_and_positive_grasp_zone_unchanged': True}
    (H / 'frame_design.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return R, receipt


def horizontal_material_pre(m, d, center, R, target):
    # The single material approach must follow its horizontal local Z, rather
    # than inherit the old world-down descent. Use the full actual payload and
    # open-width-independent hand/finger envelope to derive one outside point.
    groups = {}
    ep = d.site_xpos[center].copy()
    for entity, prefix in [('payload', 'fork_bowl_1_'), ('hand', 'gripper0_')]:
        points = []
        for g in range(m.ngeom):
            if not ((m.geom(g).name or '').startswith(prefix)
                    and (m.geom_contype[g] or m.geom_conaffinity[g])):
                continue
            G = d.geom_xmat[g].reshape(3, 3)
            a = m.geom_aabb[g]
            points.extend(d.geom_xpos[g] + G @ (a[:3] + a[3:] * np.asarray(s))
                          for s in itertools.product((-1, 1), repeat=3))
        if not points:
            raise RuntimeError('Missing full tangent pre-approach geometry ' + entity)
        groups[entity] = np.asarray(points)
    approach = R[:, 2]
    payload_front = float(np.min(groups['payload'] @ approach))
    # d has been returned to the actual saved home for continuous-route IK.
    # Transform its compiled hand vertices into EEF local space, then use that
    # same local Z extent in the new material frame; do not treat home as R.
    current_E = d.site_xmat[center].reshape(3, 3)
    hand_forward = float(np.max(((groups['hand'] - ep) @ current_E)[:, 2]))
    shift = max(0., float(target @ approach + hand_forward - payload_front + .005))
    pre = target - shift * approach
    (H / 'material_pre_design.json').write_text(json.dumps({
        'whole_payload_front_plane': payload_front,
        'full_hard_hand_finger_forward_extent': hand_forward,
        'ordinary_geometric_outside_clearance_m': .005,
        'unique_approach_distance_m': shift, 'fixed_pre_target': pre.tolist(),
        'approach_axis_world': approach.tolist(), 'no_position_grid': True,
        'above_height_inherited_from_original_route_m': 1.15}, indent=2) + '\n')
    return pre


def selected():
    src = textwrap.dedent(inspect.getsource(parent.main))
    old = '        down = np.array([0.,0.,-1.]); R = np.column_stack((radial,np.cross(down,radial),down))'
    if src.count(old) != 1:
        raise RuntimeError('Actual original material-frame source count changed')
    src = src.replace(old,
        "        R, frame_receipt = tangent_material_frame(m, d, center, radial)\n"
        "        down = R[:, 2].copy()\n"
        "        result['tangent_material_frame'] = frame_receipt")
    old_route = "        above=target.copy();above[2]=1.15\n        segment('radial_above_original_bowl',above)\n        segment('radial_vertical_material_approach',target)"
    if src.count(old_route) != 1:
        raise RuntimeError('Actual original source approach source count changed')
    src = src.replace(old_route,
        "        pre = horizontal_material_pre(m, d, center, R, target)\n"
        "        above=pre.copy();above[2]=1.15\n"
        "        segment('tangent_above_outside_original_bowl',above)\n"
        "        segment('tangent_lower_outside_original_bowl',pre)\n"
        "        segment('tangent_horizontal_material_approach',target)")
    ns = dict(parent.main.__globals__)
    ns.update(H=H, SAVED=SAVED, ROOT=ROOT, tangent_material_frame=tangent_material_frame,
              horizontal_material_pre=horizontal_material_pre)
    exec(compile(src, __file__ + '::one-derived-horizontal-material-frame', 'exec'), ns)
    return ns['main'], src


def main():
    selected()[0]()


if __name__ == '__main__':
    main()
