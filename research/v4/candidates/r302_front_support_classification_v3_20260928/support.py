"""Exact-pair offline classification; raw physics and hard guards retained."""
import inspect
import itertools
import math
from pathlib import Path

import mujoco
import numpy as np

from research.v4.candidates.r302_front_pair_contact_20260928.scene import PARAMS
from research.v4.candidates.r302_front_slot_short_v_20260928 import support as original
from research.v4.candidates.r302_horizontal_end_grasp_saved_copy_20260928.derive import CAN_NAMES

H = Path(__file__).resolve().parent


def allowed_pairs(model):
    """Fail closed on IDs, explicit pair entries and every effective field."""
    if model.npair != 2:
        raise RuntimeError('Only the two exact installed source pairs may be classified')
    result = {}
    for fields in PARAMS['rows']:
        can, rail = int(model.geom(fields['can']).id), int(model.geom(fields['rail']).id)
        name = 'r302_source_pair_' + fields['rail'].rsplit('_', 1)[-1]
        index = int(model.pair(name).id)
        if index < 0 or (int(model.pair_geom1[index]), int(model.pair_geom2[index])) != (can, rail):
            raise RuntimeError('Exact source support pair name/geom-ID mismatch')
        expected = {'dim': fields['dim'], 'solref': fields['new_solref'],
                    'solreffriction': fields['old_solreffriction'], 'solimp': fields['solimp'],
                    'friction': fields['friction'], 'margin': fields['margin'], 'gap': fields['gap']}
        for key, value in expected.items():
            actual = getattr(model, 'pair_' + key)[index]
            if not np.array_equal(np.asarray(actual), np.asarray(value)):
                raise RuntimeError('Source-support pair effective field changed: ' + key)
        result[frozenset((can, rail))] = {'explicit_pair_id': index, 'explicit_pair_name': name,
                                        'can_geom_id': can, 'rail_geom_id': rail,
                                        'effective_fields': expected}
    if len(result) != 2:
        raise RuntimeError('Duplicate or partial source-support pair classification')
    return result


def classified_contacts(model, data):
    pairs = allowed_pairs(model)
    rows = original.exact_contacts(model, data)
    for row in rows:
        ids = [int(model.geom(n).id) for n in row['geoms']]
        pair = pairs.get(frozenset(ids))
        row.update(geom_ids=ids, exact_source_support_pair=pair is not None,
                   offline_classification='support_compression' if pair and row['dist_m'] < 0
                   else 'source_support_contact' if pair else 'original_non_support_contact')
        if pair:
            row['explicit_source_pair'] = pair
    return rows


def hard_bad(row):
    return (row['dist_m'] < -.001 and not row['exact_two_soft_pad_can_material']
            and not row['exact_source_support_pair'])


def cylinder_v_bounds(model, data, ids, V, origin):
    bounds = []
    for i in ids:
        if int(model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            raise RuntimeError('Own can collision geometry changed from analytic cylinders')
        axis = data.geom_xmat[i].reshape(3, 3)[:, 2]
        center = V.T @ (data.geom_xpos[i] - origin)
        spans = [float(model.geom_size[i, 0] * math.sqrt(max(0., 1 - float(v @ axis) ** 2))
                       + model.geom_size[i, 1] * abs(float(v @ axis))) for v in V.T]
        bounds.append({'geom_id': i, 'geom': model.geom(i).name,
                       'V_local_bounds_m': np.column_stack((center - spans, center + spans)).tolist()})
    return bounds


def enrich(scene, row):
    m, d = scene.backend.sim.model._model, scene.backend.sim.data._data
    pairs = allowed_pairs(m)
    can_ids = {int(m.geom(n).id) for n in CAN_NAMES}
    support = []
    world_total = np.zeros(3)
    all_can_world = np.zeros(3)
    for index, contact in enumerate(d.contact[:d.ncon]):
        ids = [int(contact.geom1), int(contact.geom2)]
        pair = pairs.get(frozenset(ids))
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, index, force)
        sign = -1 if ids[0] in can_ids else 1 if ids[1] in can_ids else 0
        on_can = sign * np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
        all_can_world += on_can
        if pair:
            world_total += on_can
            support.append({'contact_id': index, 'geom_ids': ids,
                            'geoms': [m.geom(i).name for i in ids],
                            'explicit_source_pair': pair, 'dist_m': float(contact.dist),
                            'offline_classification': 'support_compression' if contact.dist < 0 else 'source_support_contact',
                            'world_position_m': contact.pos.tolist(),
                            'contact_frame_world': contact.frame.tolist(),
                            'normal_on_can_world_unit': (sign * np.asarray(contact.frame[:3])).tolist(),
                            'force_contact_frame_6D': force.tolist(),
                            'world_force_on_can_N': on_can.tolist(),
                            'actual_contact_dim': int(contact.dim),
                            'actual_contact_solref': contact.solref.tolist(),
                            'actual_contact_solreffriction': contact.solreffriction.tolist(),
                            'actual_contact_solimp': contact.solimp.tolist(),
                            'actual_contact_friction': contact.friction.tolist(),
                            'actual_contact_includemargin_m': float(contact.includemargin)})
    base = int(m.geom('r302_v_base').id)
    V, origin = d.geom_xmat[base].reshape(3, 3), d.geom_xpos[base]
    local_com = V.T @ (np.asarray(row['can_com_world']) - origin)
    margin = m.geom_size[base, :2] - np.abs(local_com[:2])
    boxes = []
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    for name in ('r302_v_base', 'r302_v_rail_-1', 'r302_v_rail_1'):
        i = int(m.geom(name).id)
        if int(m.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise RuntimeError('Actual V support box type changed')
        vertices = signs * m.geom_size[i] @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]
        boxes.append({'geom_id': i, 'geom': name, 'actual_world_vertices_m': vertices.tolist()})
    contacts = row['whole_robot_can_environment_contacts']
    raw_bad = [r for r in contacts if r['dist_m'] < -.001 and not r['exact_two_soft_pad_can_material']]
    remaining_bad = [r for r in contacts if hard_bad(r)]
    row.update(actual_full_qpos=d.qpos.tolist(), actual_full_qvel=d.qvel.tolist(),
               actual_full_ctrl=d.ctrl.tolist(), actual_source_support_contacts=support,
               exact_source_pair_total_world_load_N=world_total.tolist(),
               actual_all_can_world_load_N=all_can_world.tolist(),
               support_load_Z_minus_constraint_Z_N=float(all_can_world[2] - row['can_vertical_constraint_force_n']),
               original_blanket_nonpad_gt1mm_guard_passed=not raw_bad,
               original_blanket_guard_raw_failures=raw_bad,
               exact_support_classified_hard_guard_passed=not remaining_bad,
               actual_assist_held=bool(scene.grasp_assist.held),
               support_envelope={'actual_V_base_world_center_m': origin.tolist(), 'actual_V_R': V.tolist(),
                    'actual_V_base_half_xy_m': m.geom_size[base, :2].tolist(),
                    'actual_can_COM_in_V_frame_m': local_com.tolist(),
                    'actual_COM_base_XY_margin_m': margin.tolist(),
                    'actual_COM_projection_inside_base': bool(np.all(margin >= 0)),
                    'can_full_collision_bounds_in_V_frame': cylinder_v_bounds(m, d, sorted(can_ids), V, origin),
                    'actual_support_box_vertices': boxes,
                    'not_a_full_can_footprint_requirement': True,
                    'source_negative_distance_not_hidden_or_claimed_no_penetration': True})


# One exact, logged observation-only transform of the frozen sampler.
SOURCE = inspect.getsource(original.observed_sample)
OLD = "and not r['exact_two_soft_pad_can_material']]"
NEW = "and not r['exact_two_soft_pad_can_material'] and not r['exact_source_support_pair']]"
INSERT = "    row['actual_all_contact_forces']=forces\n"
if SOURCE.count(OLD) != 1 or SOURCE.count(INSERT) != 1:
    raise RuntimeError('Frozen sampler classification seam changed; STOP')
EXECUTED_SOURCE = SOURCE.replace(OLD, NEW).replace(INSERT, INSERT + '    enrich(scene,row)\n')
namespace = dict(original.observed_sample.__globals__, H=H,
                 exact_contacts=classified_contacts, enrich=enrich)
exec(compile(EXECUTED_SOURCE, str(H / 'executed_sampler.py.txt'), 'exec'), namespace)
sample = namespace['observed_sample']
