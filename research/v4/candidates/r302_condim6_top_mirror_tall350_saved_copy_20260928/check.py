"""One own compile, fixed65 full-contact query, then original copied route."""
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from research.v4.candidates.r302_condim6_top_mirror_source8mm_saved_copy_20260928 import check as previous
from research.v4.candidates.r302_horizontal_end_grasp_saved_copy_20260928 import derive
from .scene import H, CompileOnlyTall350CanScene

ROOT = H.parents[3]
COMPILED = H / 'compiled_seed0'
SAVED = previous.mirror.parent.SAVED


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def admit():
    f = json.loads((H / 'source_freeze_before.json').read_text())
    drift = [p for p, s in f['source_sha256'].items() if not (ROOT / p).is_file() or sha(ROOT / p) != s]
    if drift:
        raise RuntimeError('Exact own qualified-device source admission failed: ' + repr(drift))
    return len(f['source_sha256'])


def model_guard(old, new):
    out = dict(passed=False, failures=[], changed_model_arrays=[], non_device_geoms=0,
               original_heating_region_unchanged=False, new_physical_device_not_equivalent=True)
    dimensions = ('nq', 'nv', 'nbody', 'njnt', 'ngeom', 'nsite', 'ncam', 'nu', 'nsensor', 'neq', 'npair', 'nmesh')
    out['dimensions'] = {n: [int(getattr(old, n)), int(getattr(new, n))] for n in dimensions}
    for n, pair in out['dimensions'].items():
        if pair[0] != pair[1]: out['failures'].append('dimension:' + n)
    for n in dir(old):
        a, b = getattr(old, n, None), getattr(new, n, None)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and not (a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()):
            out['changed_model_arrays'].append(n)
    for i in range(old.ngeom):
        name = old.geom(i).name
        if name != new.geom(i).name: out['failures'].append('geom_ID_name:' + str(i)); continue
        for field in ('geom_contype', 'geom_conaffinity', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap', 'geom_group'):
            if not np.array_equal(getattr(old, field)[i], getattr(new, field)[i]): out['failures'].append(name + '/' + field)
        if name.startswith('microwave_1_'): continue
        out['non_device_geoms'] += 1
        for field in ('geom_type', 'geom_pos', 'geom_quat', 'geom_size', 'geom_rgba', 'geom_bodyid', 'geom_dataid'):
            if not np.array_equal(getattr(old, field)[i], getattr(new, field)[i]): out['failures'].append(name + '/' + field)
    for i in range(old.nbody):
        name = old.body(i).name
        if name != new.body(i).name: out['failures'].append('body_ID_name:' + str(i)); continue
        if name.startswith('microwave_1_'): continue
        for field in ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'body_pos', 'body_quat'):
            if not np.array_equal(getattr(old, field)[i], getattr(new, field)[i]): out['failures'].append(name + '/' + field)
    fields = ('cam_pos', 'cam_quat', 'cam_fovy', 'cam_bodyid', 'jnt_range', 'jnt_qposadr', 'jnt_dofadr',
              'actuator_ctrlrange', 'dof_damping', 'eq_data', 'eq_active0', 'eq_obj1id', 'eq_obj2id',
              'pair_geom1', 'pair_geom2', 'pair_dim', 'pair_solref', 'pair_solreffriction', 'pair_solimp',
              'pair_friction', 'pair_margin', 'pair_gap')
    for field in fields:
        if not np.array_equal(getattr(old, field), getattr(new, field)): out['failures'].append(field)
    if old.opt.timestep != new.opt.timestep or not np.array_equal(old.opt.gravity, new.opt.gravity):
        out['failures'].append('timestep/gravity')
    for name in ('microwave_1_heating_region', 'v42_steel_thermos_1_grasp_zone'):
        a, b = old.site(name).id, new.site(name).id
        for field in ('site_pos', 'site_quat', 'site_size', 'site_bodyid'):
            if not np.array_equal(getattr(old, field)[a], getattr(new, field)[b]): out['failures'].append(name + '/' + field)
    i = old.geom('microwave_1_g7').id
    for field in ('geom_pos', 'geom_quat', 'geom_size'):
        if not np.array_equal(getattr(old, field)[i], getattr(new, field)[i]): out['failures'].append('original_floor/' + field)
    out['original_heating_region_unchanged'] = not any('heating_region' in n for n in out['failures'])
    out['device_body_changes'] = [dict(name=old.body(i).name, old_mass=float(old.body_mass[i]), new_mass=float(new.body_mass[i]),
        old_ipos=old.body_ipos[i].tolist(), new_ipos=new.body_ipos[i].tolist(),
        old_inertia=old.body_inertia[i].tolist(), new_inertia=new.body_inertia[i].tolist()) for i in range(old.nbody) if old.body(i).name.startswith('microwave_1_')]
    out['passed'] = not out['failures']
    return out


def all_contacts(m, d):
    cans = {m.geom(n).id for n in derive.CAN_NAMES}; pads = {m.geom(n).id for n in derive.PAD_NAMES}
    pairs = {frozenset((m.geom(derive.CAN_NAMES[0]).id, m.geom('r302_v_rail_' + sign).id)) for sign in ('-1', '1')}
    rows = []
    for c in d.contact[:d.ncon]:
        ids = [int(c.geom1), int(c.geom2)]; names = [m.geom(i).name or '' for i in ids]
        if not any(n.startswith(('robot0_', 'gripper0_', 'steel_thermos_1_', 'fixed_mount0_')) for n in names): continue
        soft = any(i in pads for i in ids) and any(i in cans for i in ids)
        support = frozenset(ids) in pairs
        rows.append(dict(ids=ids, names=names, dist_m=float(c.dist), position_m=c.pos.tolist(), frame=c.frame.tolist(),
            exact_pad_can=soft, exact_source_support_compression=support,
            hard_stop=bool(c.dist < -.001 and not soft and not support), not_physical_force_evidence=True))
    return rows


def fixed65():
    m = mujoco.MjModel.from_binary_path(str(COMPILED / 'compiled_model.mjb'))
    old = np.load(SAVED / 'constructor_data.npz', allow_pickle=False)
    rows = json.loads((previous.H / 'copied_report.json').read_text())['rows']
    if len(rows) != 65: raise RuntimeError('Original65 saved pose receipt differs')
    d = mujoco.MjData(m); e = m.site('gripper0_right_grip_site').id
    arms = m.jnt_qposadr[[m.joint('robot0_joint' + str(i)).id for i in range(1, 8)]]
    fingers = [int(m.jnt_qposadr[m.joint(n).id]) for n in ('gripper0_right_finger_joint1', 'gripper0_right_finger_joint2')]
    ca = int(m.jnt_qposadr[m.joint('steel_thermos_1_joint0').id])
    out = dict(kind='old65 complete pose reconstruction in actual new model', passed=False, failure=None,
               IK_calls=0, actual_controls=0, rows=[])
    try:
        for k, row in enumerate(rows, 1):
            d.qpos[:] = old['qpos']; d.qvel[:] = 0.
            d.qpos[arms] = row['joint_limit_margins']['qpos_rad']
            d.qpos[fingers] = row['finger_qpos_m']; d.qpos[ca:ca + 7] = row['can_qpos']
            mujoco.mj_forward(m, d)
            cs = all_contacts(m, d)
            result = dict(row=k, stage=row['stage'], qpos=d.qpos.tolist(), actual_eef_m=d.site_xpos[e].tolist(),
                saved_eef_byteexact=np.array_equal(d.site_xpos[e], row['actual_eef_m']), contacts=cs)
            out['rows'].append(result)
            if not result['saved_eef_byteexact']: raise RuntimeError('First changed original saved robot FK at row' + str(k))
            if any(c['hard_stop'] for c in cs): raise RuntimeError('First other full saved-pose hard collision at row' + str(k))
        out['passed'] = True
    except Exception as exc:
        out['failure'] = dict(type=type(exc).__name__, message=str(exc))
    (H / 'fixed65_report.json').write_text(json.dumps(out, indent=2) + '\n')
    return out['passed']


def selected_route():
    _, _, original = previous.selected()
    before = "    m = mujoco.MjModel.from_binary_path(str(SAVED / 'compiled_model.mjb'))\n"
    after = "    m = mujoco.MjModel.from_binary_path(str(H / 'compiled_seed0/compiled_model.mjb'))\n"
    if original.count(before) != 1: raise RuntimeError('Own actual model-loader seam changed')
    changed = original.replace(before, after)
    if changed.replace(after, before) != original: raise RuntimeError('Only qualified model input may change')
    ns = dict(previous.mirror.parent.main.__globals__, H=H, admit=admit)
    exec(compile(changed, str(H / 'selected_parent_main.py.txt'), 'exec'), ns)
    (H / 'selected_parent_main.py.txt').write_text(changed)
    return ns['main'], dict(original_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        selected_source_sha256=hashlib.sha256(changed.encode()).hexdigest(), model_input_only_inverse_exact=True,
        same_original_ik=ns['_ik'] is previous.mirror.parent._ik, same_actual80_data_input=str(SAVED),
        source8mm_only_retained=True, all_transport_defaults_unchanged=True, target_waypoint_not_extended=True)


def main():
    admit()
    if COMPILED.exists(): raise RuntimeError('Preserve one new device compile; no repeat')
    COMPILED.mkdir(); started = time.monotonic(); s = None
    receipt = dict(passed=False, failure=None, PID=os.getpid(), Scene_constructors=1,
                   real_controls=0, startup80_rerun=False, GPU_services_models=0)
    try:
        s = CompileOnlyTall350CanScene(0, camera_observations=False)
        if s.sequence != 0: raise RuntimeError('Compile-only constructor advanced real controls')
        m, d = s.backend.sim.model._model, s.backend.sim.data._data
        (COMPILED / 'scene.xml').write_text(s.backend.sim.model.get_xml())
        mujoco.mj_saveModel(m, str(COMPILED / 'compiled_model.mjb'))
        np.savez_compressed(COMPILED / 'constructor_data.npz', qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), act=d.act.copy(), time=np.asarray(d.time))
        guard = model_guard(mujoco.MjModel.from_binary_path(str(SAVED / 'compiled_model.mjb')), m)
        (H / 'compiled_device_guard.json').write_text(json.dumps(guard, indent=2) + '\n')
        receipt.update(precompile_receipt=s.backend.r302_tall350_precompile_receipt,
            initial_contacts=all_contacts(m, d), guard_passed=guard['passed'], sequence=s.sequence)
        if not guard['passed']: raise RuntimeError('First unexpected own physical/scorer/controller difference: ' + repr(guard['failures']))
        if any(c['hard_stop'] for c in receipt['initial_contacts']): raise RuntimeError('First new initial hard collision')
        receipt['passed'] = True
    except BaseException as exc:
        receipt['failure'] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        if s is not None: s.close()
        receipt['elapsed_s'] = time.monotonic() - started
        (H / 'compile_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not receipt['passed']:
        print(json.dumps(dict(compile_passed=False, failure=receipt['failure']))); return
    if not fixed65():
        print(json.dumps(dict(compile_passed=True, fixed65_passed=False))); return
    target, binding = selected_route()
    (H / 'binding_receipt.json').write_text(json.dumps(binding, indent=2) + '\n')
    target()


if __name__ == '__main__':
    main()
