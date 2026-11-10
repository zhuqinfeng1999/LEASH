"""One jaw-symmetric frame on actual saved seed2 reset; no live Scene."""
from pathlib import Path
from types import FunctionType
import hashlib
import json
import time

import mujoco
import numpy as np

from research.v4.candidates.r305_robot_facing_lug_y80_copy_20260928 import check as parent
from research.v4.candidates.r305_horizontal_lug_y80_static_20260928 import check as material
from research.v4.candidates.r305_tall_microwave_single_lug_20260928.check import full_contacts

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
MODEL = H.parent / 'r305_robot_facing_lug_y80_copy_20260928/actual_constructor_seed1/actual_model.mjb'
FAIL = H.parent / 'r305_side_actual_qualification_20260928/root_actual_seeds0_9/naive_seed2.json'
ORIGINAL_IK = parent.lug.ORIGINAL_IK
IK_LOG = []


def mirrored_material(m, d, center, body, left, right, finger):
    original_R, old_target, closed, design = material.side_material(m, d, center, body, left, right, finger)
    midpoint = np.asarray(design['pad_midpoint_EEF_local'])
    material_center = old_target + original_R @ midpoint
    R = original_R.copy()
    R[:, :2] *= -1
    target = material_center - R @ midpoint
    if not np.array_equal(R[:, 2], original_R[:, 2]):
        raise RuntimeError('Material inward Z changed')
    if not np.allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-12) or np.linalg.det(R) < .999999:
        raise RuntimeError('Jaw mirror is not the original equivalent right-handed frame')
    result = dict(design, version='R305-single-inwardZ-jawXY-mirror-v1',
                  original_R=original_R.tolist(), target_R=R.tolist(),
                  original_target_EEF=old_target.tolist(), target_EEF=target.tolist(),
                  material_center_unchanged=material_center.tolist(),
                  same_inward_Z=True, same_material_width_and_closed_q=True,
                  jaw_X_and_Y_exact_negative=True, no_angle_position_height_grid=True)
    return R, target, closed, result


def joint_rows(m, d):
    rows = []
    for i in range(1, 8):
        j = m.joint('robot0_joint' + str(i)).id
        q = float(d.qpos[m.jnt_qposadr[j]])
        lo, hi = map(float, m.jnt_range[j])
        rows.append(dict(name=m.joint(j).name, qpos_rad=q, range_rad=[lo, hi],
                         lower_margin_rad=q-lo, upper_margin_rad=hi-q,
                         legal=bool(lo <= q <= hi)))
    return rows


def logged_ik(proxy, target, R, **kwargs):
    answer = ORIGINAL_IK(proxy, target, R, **kwargs)
    m = proxy.model._model
    d = mujoco.MjData(m)
    d.qpos[:] = proxy.data.qpos
    if answer.get('reachable'):
        for i, q in enumerate(answer['joint_qpos_rad'], 1):
            d.qpos[m.jnt_qposadr[m.joint('robot0_joint'+str(i)).id]] = q
    mujoco.mj_forward(m, d)
    center = m.site('gripper0_right_grip_site').id
    row = dict(index=len(IK_LOG), actual_return=answer, target=np.asarray(target).tolist(),
               R=np.asarray(R).tolist(), original_parameters=kwargs,
               qpos=d.qpos.tolist(), EEF=d.site_xpos[center].tolist(),
               EEF_R=d.site_xmat[center].reshape(3,3).tolist(),
               joint_limits=joint_rows(m,d), contacts=full_contacts(m,d),
               scope='Independent copied IK diagnostic, no actual robot control/contact force')
    IK_LOG.append(row)
    (H / 'IK_actual_returns.json').write_text(json.dumps(IK_LOG, indent=2) + '\n')
    return answer


def selected():
    original, source = parent.selected()
    old = "m = mujoco.MjModel.from_binary_path(str(SAVED / 'lug_model.mjb'))"
    if source.count(old) != 1:
        raise RuntimeError('Exact saved-model route load seam changed')
    source = source.replace(old, 'm = mujoco.MjModel.from_binary_path(str(MODEL))')
    pre = original.__globals__['horizontal_material_pre']
    pre_ns = pre.__globals__ | {'H': H}
    pre_copy = FunctionType(pre.__code__, pre_ns, pre.__name__, pre.__defaults__, pre.__closure__)
    ns = original.__globals__ | dict(H=H, ROOT=ROOT, MODEL=MODEL,
                                     lug_material=mirrored_material, _ik=logged_ik,
                                     horizontal_material_pre=pre_copy, contact_rows=full_contacts)
    exec(compile(source, str(H / 'selected_copy_source.py'), 'exec'), ns)
    return ns['main'], source


def main():
    if (H / 'report.json').exists() or (H / 'input_contract.json').exists():
        raise RuntimeError('One authorized fixed mirrored observation only')
    pins = json.loads((H / 'source_freeze.json').read_text())
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source drift ' + name)
    actual = json.loads(FAIL.read_text())
    m = mujoco.MjModel.from_binary_path(str(MODEL))
    saved = np.load(H / 'robot_facing_copy_input.npz')
    d = mujoco.MjData(m)
    d.qpos[:] = saved['qpos']
    d.qvel[:] = saved['qvel']
    d.time = actual['common_reset_trace'][-1]['time']
    mujoco.mj_forward(m, d)
    body = m.body('fork_bowl_1_main').id
    zone = m.site('v42_fork_bowl_1_grasp_zone').id
    oven = m.body('microwave_1_main').id
    center = m.site('gripper0_right_grip_site').id
    left, right = m.site('v42_left_pad').id, m.site('v42_right_pad').id
    finger = [m.joint('gripper0_right_finger_joint1').id, m.joint('gripper0_right_finger_joint2').id]
    R, target, closed, design = mirrored_material(m,d,center,body,left,right,finger)
    reference = actual['actual_private_lug_design']
    report = dict(passed=False, actual_saved_seed=2, detached_reference_model_seed=1,
                  model_compiles_or_Scene=0, actual_task_or_reset_controls=0, GPU_models_services=0,
                  initial_qpos_exact=np.array_equal(d.qpos, actual['actual_initial_qpos']),
                  initial_qvel_exact=np.array_equal(d.qvel, actual['actual_initial_qvel']),
                  payload_world_exact=np.array_equal(d.xpos[body], actual['initial_truth']['metal_item_center_m']),
                  initial_payload=d.xpos[body].tolist(), initial_oven=d.xpos[oven].tolist(),
                  body_mass_kg=float(m.body_mass[body]), zone_local=m.site_pos[zone].tolist(),
                  robot_base=m.body_pos[m.body('robot0_base').id].tolist(),
                  actual_arm_names_qpos_addresses=[dict(name=m.joint('robot0_joint'+str(i)).name,
                        address=int(m.jnt_qposadr[m.joint('robot0_joint'+str(i)).id]))for i in range(1,8)],
                  reference_design_R_exact=np.array_equal(design['original_R'], reference['target_R']),
                  reference_design_target_delta_m=(np.asarray(design['original_target_EEF'])-reference['target_EEF']).tolist(),
                  design=design, initial_contacts=full_contacts(m,d), joint_limits=joint_rows(m,d),
                  active_equalities=[dict(name=m.equality(i).name, active=bool(d.eq_active[i]))for i in range(m.neq)],
                  limitation='Saved detached MJB is own seed1 reference, actual seed2 after20 qpos/qvel restored only to independent data. No whole live cache/state or exact seed2 compiled-model equivalence claim.')
    if not all(report[k]for k in ('initial_qpos_exact','initial_qvel_exact','payload_world_exact','reference_design_R_exact')):
        report['failure'] = 'Actual seed2 saved input/design mapping mismatch'
    elif any(c['hard_stop']for c in report['initial_contacts']):
        report['failure'] = 'Initial copied hard collision >1mm'
    elif report['body_mass_kg'] != .31 or report['zone_local'] != [.08,0.,.022]:
        report['failure'] = 'Original body/material values changed'
    else:
        report['passed'] = True
    (H / 'input_contract.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError(report['failure'])
    callback, source = selected()
    if (H / 'selected_copy_source.txt').read_text() != source:
        raise RuntimeError('Frozen selected numerical route source differs')
    callback()


if __name__ == '__main__':
    main()
