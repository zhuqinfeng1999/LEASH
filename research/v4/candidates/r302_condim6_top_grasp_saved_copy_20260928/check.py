"""ONE actual saved80 geometry/top-route query, zero Scene or real controls."""
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from research.v4.accelerated.slot_preflight import _ik
from research.v4.candidates.r302_horizontal_end_grasp_saved_copy_20260928 import derive
from research.v4.candidates.r302_horizontal_short_v_20260928.copy import shape_interval, box_vertices
from research.v4.candidates.r3_04_source_axis_mirrored_low_copy_20260928.check import joint_margins

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
PARENT = ROOT / 'research/v4/candidates/r302_source_pair_condim6_20260928'
SAVED = PARENT / 'startup_seed0'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def admit():
    f = json.loads((H / 'source_freeze_before.json').read_text())
    drift = [p for p, s in f['source_sha256'].items() if not (ROOT / p).is_file() or sha(ROOT / p) != s]
    if drift:
        raise RuntimeError('Exact source admission failed: ' + repr(drift))
    return len(f['source_sha256'])


def main():
    if (H / 'geometry.json').exists() or (H / 'copied_report.json').exists():
        raise RuntimeError('Preserve first top-grasp result; no repeat')
    count = admit()
    start = time.monotonic()
    report = dict(kind='one saved80 normal top-grasp geometry and continuous route',
                  source_count=count, Scene_constructors=0, physics_steps=0,
                  actual_controls=0, services_GPU_models=0, passed=False,
                  failure=None, rows=[], virtual_relative_private_offline_only=True,
                  actual_bind_lift_or_method_claim=False, original_caps=[1200, 90])
    m = mujoco.MjModel.from_binary_path(str(SAVED / 'compiled_model.mjb'))
    old = np.load(SAVED / 'constructor_data.npz', allow_pickle=False)
    d = mujoco.MjData(m)
    d.qpos[:] = old['qpos']; d.qvel[:] = old['qvel']
    d.ctrl[:] = old['ctrl']; d.act[:] = old['act']; d.time = float(old['time'])
    mujoco.mj_forward(m, d)
    e = m.site('gripper0_right_grip_site').id
    zone = m.site('v42_steel_thermos_1_grasp_zone').id
    body = m.body('steel_thermos_1_main').id
    oven = m.body('microwave_1_main').id
    ca = int(m.jnt_qposadr[m.joint('steel_thermos_1_joint0').id])
    arms = [m.joint('robot0_joint' + str(i)).id for i in range(1, 8)]
    adr = m.jnt_qposadr[arms]
    f = [int(m.jnt_qposadr[m.joint(n).id]) for n in (
        'gripper0_right_finger_joint1', 'gripper0_right_finger_joint2')]
    zone_p = d.site_xpos[zone].copy()
    B = d.xmat[body].reshape(3, 3).copy()
    can_original = d.qpos[ca:ca + 7].copy()
    if not np.array_equal(m.site_pos[zone], [0, 0, .042]) or abs(m.body_mass[body] - .4) > 1e-12:
        raise RuntimeError('Own original can/zone identity mismatch')
    # Actual EEF local X is the closing axis. Local Y follows the can's
    # projected long axis; local Z is world-down. This is one signed frame.
    long = B[:, 2].copy(); long[2] = 0.; long /= np.linalg.norm(long)
    down = np.asarray([0., 0., -1.])
    Q = np.column_stack((np.cross(long, down), long, down))
    if not np.allclose(Q.T @ Q, np.eye(3), atol=1e-12) or np.linalg.det(Q) < .999999:
        raise RuntimeError('Derived top frame not proper SO(3)')
    cans = [m.geom(n).id for n in derive.CAN_NAMES]
    qclose = float(m.geom_size[cans[0], 0]) + .0005
    pads = [m.geom(n).id for n in derive.PAD_NAMES]
    for i in pads:
        if int(m.geom_type[i]) != 6 or not np.array_equal(m.geom_size[i], [.008, .004, .008]) or not np.array_equal(m.geom_solref[i], [.01, .5]):
            raise RuntimeError('Exact actual soft-pad metadata mismatch')
    # The existing source pair is the only permitted source-compression
    # classification. It never exempts hand/wood, hand/table or can/oven.
    support_pairs = {frozenset((cans[0], m.geom('r302_v_rail_' + sign).id)) for sign in ('-1', '1')}
    pair_ids = [m.pair('r302_source_pair_' + sign).id for sign in ('-1', '1')]
    if any(m.pair_dim[i] != 6 for i in pair_ids):
        raise RuntimeError('Actual condim6 source pair mismatch')
    d.qpos[f] = [.04, -.04]; mujoco.mj_forward(m, d)
    hand = derive.hand_geometry(m, d)
    palm = next(r for r in hand if r['name'] == 'gripper0_right_hand_collision')
    palm_low_z = float((np.asarray(palm['vertices_eef']) @ Q.T)[:, 2].min())
    can_upper = shape_interval(m, d, cans, np.asarray([0., 0., 1.]), zone_p)[1]
    backoff = can_upper + .005 - palm_low_z
    closure = zone_p + [0., 0., backoff]
    if not 0 < backoff <= .020:
        raise RuntimeError('Derived physical palm clearance exceeds unchanged20mm zone tolerance')
    home = d.site_xpos[e].copy(); homeQ = d.site_xmat[e].reshape(3, 3).copy()
    geom = dict(kind='one analytic material-axis top grasp; independent saved-MJB query',
                passed=False, failure=None, base_world_m=d.xpos[m.body('robot0_base').id].tolist(),
                expected_base_wording_correction_m=-.61, source_can_qpos=can_original.tolist(),
                can_axis_world=B[:, 2].tolist(), top_R=Q.tolist(), source_zone_m=zone_p.tolist(),
                unchanged_zone_local_m=m.site_pos[zone].tolist(), target_eef_m=closure.tolist(),
                derived_palm_backoff_m=backoff, palm_support_low_z_m=palm_low_z,
                full_can_top_relative_zone_m=can_upper, palm_can_vertical_gap_m=.005,
                original_tolerance_m=[.015, .015, .020], q_closed_m=qclose, pairs=[],
                pad_metadata=[dict(name=m.geom(i).name, size_m=m.geom_size[i].tolist(),
                    solref=m.geom_solref[i].tolist(), masks=[int(m.geom_contype[i]), int(m.geom_conaffinity[i])]) for i in pads])
    try:
        env = [i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and
               ((m.geom(i).name or '').startswith(('r302_v_', 'microwave_1_')) or m.geom(i).name == 'table_collision')]
        for stage, fq in [('open', .04), ('material_closed', qclose)]:
            d.qpos[f] = [fq, -fq]; mujoco.mj_forward(m, d)
            shapes = derive.hand_geometry(m, d)
            pp = [r for r in shapes if r['name'] in derive.PAD_NAMES]
            inner_gap = abs(pp[1]['local_position'][0] - pp[0]['local_position'][0]) - .008
            geom[stage + '_pad_inner_gap_m'] = inner_gap
            for r in shapes:
                i = r['id']; d.geom_xpos[i] = closure + Q @ r['local_position']
                d.geom_xmat[i] = (Q @ r['local_rotation']).reshape(9)
            for r in shapes:
                for j in cans + env:
                    line = np.zeros(6)
                    dist = float(mujoco.mj_geomDistance(m, d, r['id'], j, .30, line))
                    geom['pairs'].append(dict(stage=stage, geoms=[r['name'], m.geom(j).name],
                        signed_distance_m=dist, nearest_segment_world_m=line.tolist(),
                        exact_soft_pad_can=r['id'] in pads and j in cans))
        nonpad = [r for r in geom['pairs'] if not r['exact_soft_pad_can']]
        bad = [r for r in nonpad if r['signed_distance_m'] < -.001]
        geom['min_nonpad_distance_m'] = min(r['signed_distance_m'] for r in nonpad)
        geom['zone_eef_local_m'] = (Q.T @ (zone_p - closure)).tolist()
        geom['original_tolerance_eligible'] = bool(np.all(np.abs(Q.T @ (zone_p - closure)) <= [.015, .015, .020]))
        geom['open_full_can_diameter_margin_m'] = geom['open_pad_inner_gap_m'] - .073
        geom['material_closed_pad_can_distances_m'] = [r for r in geom['pairs'] if
            r['stage'] == 'material_closed' and r['exact_soft_pad_can']]
        if bad:
            geom['failure'] = dict(kind='first hard material/fixture intersection deeper than1mm', pair=bad[0])
        elif geom['open_full_can_diameter_margin_m'] < 0 or not geom['original_tolerance_eligible']:
            geom['failure'] = dict(kind='original material zone or complete open diameter not feasible')
        geom['passed'] = geom['failure'] is None
        (H / 'geometry.json').write_text(json.dumps(derive.serial(geom), indent=2) + '\n')
        if not geom['passed']:
            report['failure'] = geom['failure']; return
        # Return to the unmodified actual saved80 source and original PARK.
        # All following FK/virtual hold writes are only independent copied data.
        d.qpos[:] = old['qpos']; d.qvel[:] = 0.; d.qpos[f] = [.04, -.04]
        mujoco.mj_forward(m, d)
        sim = SimpleNamespace(model=SimpleNamespace(_model=m), data=d)
        report.update(base_world_m=geom['base_world_m'], top_R=Q.tolist(),
            target_closure_m=closure.tolist(), source_can_qpos=can_original.tolist(),
            source_zone_m=zone_p.tolist(), home_eef_m=home.tolist(),
            actual_original_ik_defaults=str(inspect.signature(_ik)),
            ik_sha256=sha(Path(inspect.getfile(_ik))), continuous_requires_restart0=True)
        rel = None
        source_lines, source_start = inspect.getsourcelines(_ik)
        evaluated_line = source_start + next(i for i, line in enumerate(source_lines) if 'if best is None or' in line)

        def observed_original_ik(p, R):
            # Read-only line tracing saves actual original iteration q/error;
            # the function/code/defaults/branches are not rewritten.
            evaluations = []
            def trace(frame, event, arg):
                if frame.f_code is _ik.__code__ and event == 'line' and frame.f_lineno == evaluated_line:
                    loc = frame.f_locals
                    evaluations.append(dict(restart=int(loc['restart']), iteration=int(loc['iteration']),
                        qpos=loc['data'].qpos.copy(), actual_eef_m=loc['p'].copy(), actual_R=loc['R'].copy(),
                        position_error_m=float(loc['pos_error']), rotation_error_rad=float(loc['rot_error'])))
                return trace
            old_trace = sys.gettrace()
            try:
                sys.settrace(trace)
                answer = _ik(sim, np.asarray(p), R)
            finally:
                sys.settrace(old_trace)
            return answer, evaluations

        def contacts():
            rows = []
            for c in d.contact[:d.ncon]:
                ids = [int(c.geom1), int(c.geom2)]; names = [m.geom(i).name or '' for i in ids]
                if not any(n.startswith(('robot0_', 'gripper0_', 'steel_thermos_1_', 'fixed_mount0_')) for n in names):
                    continue
                soft = any(i in pads for i in ids) and any(i in cans for i in ids)
                support = frozenset(ids) in support_pairs
                rows.append(dict(ids=ids, names=names, dist_m=float(c.dist), world_position_m=c.pos.tolist(),
                    world_frame=c.frame.tolist(), exact_pad_can=soft, exact_source_support_compression=support,
                    copied_geometry_not_force_evidence=True))
            return rows

        def pose(stage, u, p, R, held=False, closed=False):
            nonlocal rel
            # No altered defaults, solver/restart set, gain, clipping or source
            # precision. The actual original100/12mm/12deg signature is used.
            ans, evaluations = observed_original_ik(p, R)
            if ans.get('reachable'):
                d.qpos[adr] = ans['joint_qpos_rad']
            elif evaluations:
                best = min(evaluations, key=lambda x: x['position_error_m'] + .06 * x['rotation_error_rad'])
                d.qpos[:] = best['qpos']
            d.qpos[f] = [qclose, -qclose] if closed else [.04, -.04]
            mujoco.mj_forward(m, d)
            actual = d.site_xpos[e].copy(); A = d.site_xmat[e].reshape(3, 3).copy()
            if held and rel is not None:
                d.qpos[ca:ca + 3] = actual + A @ rel[0]
                d.qpos[ca + 3:ca + 7] = Rotation.from_matrix(A @ rel[1]).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(m, d)
            rows = contacts()
            row = dict(stage=stage, u=float(u), goal_m=np.asarray(p).tolist(), goal_R=R.tolist(), ik=ans,
                actual_eef_m=actual.tolist(), actual_R=A.tolist(), finger_qpos_m=d.qpos[f].tolist(),
                joint_limit_margins=joint_margins(m, d, arms), can_qpos=d.qpos[ca:ca + 7].tolist(), contacts=rows)
            row['original_ik_evaluations'] = evaluations
            row['failed_qpos_is_saved_original_best_error_row_not_adopted'] = not ans.get('reachable', False)
            report['rows'].append(row)
            if not ans.get('reachable') or ans.get('restart') != 0:
                report['failure'] = dict(kind='first original IK failure/noncontinuous restart', row=row); return False
            bad = [r for r in rows if r['dist_m'] < -.001 and not r['exact_pad_can'] and not r['exact_source_support_compression']]
            if bad:
                report['failure'] = dict(kind='first hard full-arm/source/environment intersection >1mm', contact=bad[0], row=row); return False
            if stage == 'source_closure_open':
                z = d.site_xpos[zone].copy(); a = d.site_xpos[m.site('v42_left_pad').id]; b = d.site_xpos[m.site('v42_right_pad').id]
                gap = b - a; width = np.linalg.norm(gap); along = float(np.dot(z - a, gap / width)); local = A.T @ (z - actual)
                eligible = bool(0 <= along <= width and np.all(np.abs(local) <= [.015, .015, .020]))
                row.update(zone_eef_local_m=local.tolist(), zone_between_pad_sites=bool(0 <= along <= width), original_assist_eligible=eligible)
                if not eligible:
                    report['failure'] = dict(kind='actual copied +42mm region outside original between-finger tolerance', row=row); return False
            if stage == 'source_virtual_material_closed':
                rel = (A.T @ (d.xpos[body] - actual), A.T @ d.xmat[body].reshape(3, 3))
                report['private_copied_closed_relative'] = dict(position_m=rel[0].tolist(), R=rel[1].tolist(), not_real_bind=True)
            return True

        def segment(stage, a, b, A, C, held=False, closed=False):
            a, b = np.asarray(a), np.asarray(b)
            n = max(1, math.ceil(np.linalg.norm(b - a) / .020), math.ceil(Rotation.from_matrix(C @ A.T).magnitude() / math.radians(5)))
            sl = Slerp([0., 1.], Rotation.from_matrix([A, C]))
            for k in range(1, n + 1):
                u = k / n
                if not pose(stage, u, a * (1 - u) + b * u, sl(u).as_matrix(), held, closed):
                    return False
            return True

        above = closure + [0., 0., .110]
        ok = segment('home_to_top_source_pre', home, above, homeQ, Q)
        if ok: ok = segment('top_source_approach', above, closure, Q, Q)
        if ok: ok = pose('source_closure_open', 1, closure, Q)
        if ok: ok = pose('source_virtual_material_closed', 1, closure, Q, closed=True)
        if ok:
            origin = d.site_xpos[e].copy(); A = d.site_xmat[e].reshape(3, 3).copy()
            solids = cans + derive.hand_geoms(m)
            zrange = shape_interval(m, d, solids, np.asarray([0., 0., 1.]), origin)
            floor = box_vertices(m, d, m.geom('microwave_1_g7').id)[:, 2].max()
            roof = box_vertices(m, d, m.geom('microwave_1_g6').id)[:, 2].min()
            lo, hi = float(floor + .005 - zrange[0]), float(roof - .005 - zrange[1])
            carry = max(lo, float(origin[2] + .020))
            report['own_closed_hand_can_vertical_window'] = dict(floor_top_m=float(floor), roof_bottom_m=float(roof),
                EEF_z_interval_m=[lo, hi], selected_lowest_carry_m=carry, exact_cylinders_mesh_vertices=True)
            if carry > hi:
                report['failure'] = dict(kind='complete own can+closed hand floor/roof window empty'); ok = False
            if ok:
                lift = origin.copy(); lift[2] = carry
                ok = segment('private_virtual_held_low_lift', origin, lift, A, A, True, True)
            if ok:
                op = d.xpos[oven].copy(); oR = d.xmat[oven].reshape(3, 3).copy()
                xs = shape_interval(m, d, solids, oR[:, 0], d.site_xpos[e])
                wall_lo = ((box_vertices(m, d, m.geom('microwave_1_g3').id) - op) @ oR)[:, 0].max()
                wall_hi = ((box_vertices(m, d, m.geom('microwave_1_g4').id) - op) @ oR)[:, 0].min()
                interval = [float(wall_lo + .005 - xs[0]), float(wall_hi - .005 - xs[1])]
                report['own_closed_hand_can_lateral_window_m'] = interval
                if interval[0] > interval[1]:
                    report['failure'] = dict(kind='complete own can+hand lateral window empty'); ok = False
                else:
                    h = m.site('microwave_1_heating_region').id; hp = d.site_xpos[h]; hr = d.site_xmat[h].reshape(3, 3)
                    targetbody = hp - hr[:, 1] * (m.site_size[h, 1] - .016)
                    goal = targetbody - A @ rel[0]; local = oR.T @ (goal - op); local[0] = sum(interval) / 2
                    goal = op + oR @ local; goal[2] = carry
                    outside = lift + oR[:, 0] * (sum(interval) / 2 - (oR.T @ (lift - op))[0]); outside[2] = carry
                    report['first_original_critical_goal_m'] = goal.tolist()
                    ok = segment('low_outside_lateral_align', lift, outside, A, A, True, True)
                    if ok: ok = segment('horizontal_first_original_critical', outside, goal, A, A, True, True)
                    if ok:
                        heatlocal = hr.T @ (d.xpos[body] - hp)
                        report['copied_center_inside_original_region'] = bool(np.all(np.abs(heatlocal) < m.site_size[h]))
                        report['copied_heating_local_can_center_m'] = heatlocal.tolist()
        report['passed'] = bool(ok and report['failure'] is None)
    except BaseException as exc:
        report['failure'] = dict(kind='first copied instrumentation failure', exception=type(exc).__name__, message=str(exc))
    finally:
        report['elapsed_s'] = time.monotonic() - start
        report['row_count'] = len(report['rows'])
        (H / 'copied_report.json').write_text(json.dumps(derive.serial(report), indent=2) + '\n')
        np.savez_compressed(H / 'last_copied_data.npz', qpos=d.qpos.copy(), qvel=d.qvel.copy())
        print(json.dumps(derive.serial(dict(passed=report['passed'], rows=report['row_count'], failure=report['failure'], elapsed_s=report['elapsed_s']))))


if __name__ == '__main__':
    main()
