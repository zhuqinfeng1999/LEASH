"""One detached actual seed4 held-frame copy; no Scene, compile or physics."""
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from research.v4.accelerated.slot_preflight import _ik
from research.v4.candidates.r305_standard_top_copy_20260928.check import MODEL
from research.v4.candidates.r305_side_jaw_mirror_seed2_copy_20260928.check import full_contacts, joint_rows
from research.v4.candidates.r302_panel_normal_tangent_saved_copy_20260928.geometry import front_normal_then_tangent, interval

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RAW = ROOT / 'research/v4/candidates/r305_top_minimum_clearance_20260928/root_qualification_seed0_9/naive_seed4.json'


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def main():
    if (HERE / 'report.json').exists():
        raise RuntimeError('First copy is immutable; no grid or retry')
    pins = json.loads((HERE / 'source_freeze_before.json').read_text())
    for name, digest in pins.items():
        p = Path(name) if name.startswith('/') else ROOT / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Saved source drift ' + name)
    started = time.monotonic()
    raw = json.loads(RAW.read_text())
    actual = [r for r in raw['actual_control_rows'] if r['stage'] == 'radial_source_low_lift'][-1]
    m = mujoco.MjModel.from_binary_path(str(MODEL))
    d = mujoco.MjData(m)
    d.qpos[:] = actual['qpos']
    d.qvel[:] = actual['qvel']
    d.ctrl[:] = actual['ctrl']
    d.eq_active[:] = actual['eq_active']
    d.time = actual['time']
    weld = m.equality(raw['actual_first_bind']['actual_weld_name']).id
    m.eq_data[weld] = raw['actual_first_bind']['actual_weld_data']
    mujoco.mj_forward(m, d)
    eef = m.site('gripper0_right_grip_site').id
    body = m.body('fork_bowl_1_main').id
    oven = m.body('microwave_1_main').id
    arms = [m.joint('robot0_joint' + str(i)).id for i in range(1, 8)]
    arm_q = m.jnt_qposadr[arms]
    fingers = [int(m.jnt_qposadr[m.joint(n).id]) for n in ('gripper0_right_finger_joint1', 'gripper0_right_finger_joint2')]
    finger_q = d.qpos[fingers].copy()
    free = int(m.jnt_qposadr[m.joint('fork_bowl_1_joint0').id])
    start = d.site_xpos[eef].copy()
    A = d.site_xmat[eef].reshape(3, 3).copy()
    op = d.xpos[oven].copy()
    oR = d.xmat[oven].reshape(3, 3).copy()
    rel = (A.T @ (d.xpos[body] - start), A.T @ d.xmat[body].reshape(3, 3))
    solids = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and (m.geom(g).name or '').startswith(('fork_bowl_1_', 'gripper0_'))]
    left = interval(m, d, [m.geom('microwave_1_g3').id], oR[:, 0], op)[1]
    right = interval(m, d, [m.geom('microwave_1_g4').id], oR[:, 0], op)[0]
    extent = interval(m, d, solids, oR[:, 0], start)
    xwindow = [left + .005 - extent[0], right - .005 - extent[1]]
    carry = raw['stages'][-1]['goal_m'][2]
    outside = start + oR[:, 0] * (sum(xwindow) / 2 - (oR.T @ (start - op))[0])
    outside[2] = carry
    lateral = [r for r in raw['actual_control_rows'] if r['stage'] == 'radial_lowcarry_mouth_alignment']
    excursion = max(float((np.asarray(r['EEF']) - outside) @ oR[:, 1]) for r in lateral)
    plan = front_normal_then_tangent(m, d, solids, start, outside, position_stop_m=.008,
        rotation_stop_rad=.03, measured_forward_excursion_m=excursion)
    report = dict(passed=False, failure=None, kind='Own R305 seed4 actual held-frame normal/tangent copy',
        source_count=len(pins), Scene_compile_physics_controls_GPU_services_models=0,
        actual_prefix_replayed=False, actual_seed=4, detached_model_seed=1,
        no_compiled_seed4_or_live_cache_bit_identity_claim=True,
        start_sequence=actual['sequence'], original_raw_label=raw['evaluator']['label'],
        original_13_pass_14_attempt_16_unrun_unchanged=True,
        cached_fresh_EEF_delta_m=start-np.asarray(actual['EEF']),
        cached_fresh_rotation_delta_rad=Rotation.from_matrix(A @ np.asarray(actual['EEF_R']).T).magnitude(),
        mass_kg=float(m.body_mass[body]), zone_local_m=m.site_pos[m.site('v42_fork_bowl_1_grasp_zone').id],
        solid_types={m.geom(g).name: int(m.geom_type[g]) for g in solids},
        own_actual_measured_relative=dict(position=rel[0], R=rel[1]),
        copied_IK_parameters=dict(max_iter=200, position_tol=.012, rotation_tol=.03),
        actual_driver_unchanged=dict(position_stop_m=.008, rotation_stop_rad=.03, controls=1200, wall_s=90, reset_controls=20),
        start_contacts=full_contacts(m, d), plan=plan, lateral_window=xwindow, rows=[])
    proxy = SimpleNamespace(model=SimpleNamespace(_model=m), data=d)

    def segment(name, goal):
        begin = d.site_xpos[eef].copy()
        goal = np.asarray(goal)
        steps = max(1, math.ceil(np.linalg.norm(goal - begin) / .020))
        for i in range(1, steps + 1):
            p = begin + (goal - begin) * i / steps
            answer = _ik(proxy, p, A, max_iter=200, position_tol=.012, rotation_tol=.03)
            if answer.get('reachable'):
                d.qpos[arm_q] = answer['joint_qpos_rad']
            d.qpos[fingers] = finger_q
            mujoco.mj_forward(m, d)
            E = d.site_xpos[eef].copy()
            Q = d.site_xmat[eef].reshape(3, 3).copy()
            d.qpos[free:free + 3] = E + Q @ rel[0]
            d.qpos[free + 3:free + 7] = Rotation.from_matrix(Q @ rel[1]).as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(m, d)
            cs = full_contacts(m, d)
            row = dict(stage=name, fraction=i/steps, goal_m=p, ik=answer,
                actual_EEF=d.site_xpos[eef].copy(), qpos=d.qpos.copy(), joint_limits=joint_rows(m, d),
                contacts=cs, copied_geometry_not_contact_force=True)
            report['rows'].append(row)
            if not answer.get('reachable') or answer.get('restart', 0) != 0:
                raise RuntimeError('First original continuous IK gap: ' + name)
            if any(c['hard_stop'] for c in cs):
                raise RuntimeError('First non-pad hard contact >1mm: ' + name)

    try:
        if not actual['held'] or not np.allclose(start, actual['EEF'], rtol=0, atol=1e-12):
            raise RuntimeError('Actual saved held geometry mapping differs')
        if any(c['hard_stop'] for c in report['start_contacts']):
            raise RuntimeError('Actual pre-alignment start already hard')
        if xwindow[0] > xwindow[1]:
            raise RuntimeError('Original complete held mouth window empty')
        segment('front_panel_normal_retreat', plan['retreat_EEF_world_m'])
        segment('front_panel_tangent_alignment', plan['tangent_EEF_world_m'])
        hs = m.site('microwave_1_heating_region').id
        hp = d.site_xpos[hs].copy()
        hr = d.site_xmat[hs].reshape(3, 3).copy()
        local = hr.T @ (d.xpos[body] - hp)
        local[1] = -m.site_size[hs, 1] + .012
        goal = hp + hr @ local - A @ rel[0]
        goal[2] = carry
        report['original_critical_goal'] = goal
        segment('first_original_critical', goal)
        local = hr.T @ (d.xpos[body] - hp)
        report['actual_copied_center_critical'] = dict(local=local, size=m.site_size[hs],
            critical=bool(np.all(np.abs(local) < m.site_size[hs])))
        if not report['actual_copied_center_critical']['critical']:
            raise RuntimeError('Original critical not reached under copied .012')
        report['passed'] = True
    except Exception as exc:
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['elapsed_s'] = time.monotonic() - started
        (HERE / 'report.json').write_text(json.dumps(serial(report), indent=2) + '\n')
        print(json.dumps(serial(dict(passed=report['passed'], failure=report['failure'],
            rows=len(report['rows']), plan=plan, elapsed_s=report['elapsed_s'])), indent=2))


if __name__ == '__main__':
    main()
