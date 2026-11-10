"""Two fixed, model-free P201 engineering contact checks, Root-run only.

Geometry is private to this offline check. Neither case changes the assist,
controller or physical object, and no result issues EMPTY or a method label.
Importing this module constructs no scene. CPU tests use only fake scenes.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import re
import signal
import time
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
VERSION = 'p201-fixed-zone-contact-check-dev2-20261007'
WALL_S, OUTPUT_BYTES = 180, 100 * 1024**2
MOVE = dict(max_steps=100, tol=.004, step_clip=.006)
FINGER = re.compile(r'gripper0_right_finger([12])(?:_pad)?_collision$')


def registration(case):
    if case not in ('N', 'P'):
        raise ValueError('Exactly one predeclared N or P case is required')
    return dict(case=case, instance='P2-01', seed=0, version=VERSION,
        status='registered', engineering_only=True, formal=False, models=0,
        sam3=0, vla=0, online_gates=False, empty_fact_issued=False,
        closing_point_formula=('new_zone_world + object_R @ [0,0,.028]'
                               if case == 'N' else 'new_zone_world'),
        target_freeze='After the initial OPEN12, before either move',
        actions=['OPEN12', 'move source above closing point + world [0,0,.15]',
                 'move to fixed closing point', 'CLOSE12']
                 + (['one source-area lift to closing point + world [0,0,.08]']
                    if case == 'P' else []),
        move_limits=dict(MOVE), max_controls=324 if case == 'P' else 224,
        budgets=dict(wall_s=WALL_S, output_bytes=OUTPUT_BYTES))


def xyz(value):
    value = np.asarray(value, float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('A finite three-vector is required')
    return dict(zip(('x', 'y', 'z'), map(float, value)))


def fixed_targets(case, zone_world, object_rotation):
    registration(case)
    zone = np.asarray(zone_world, float)
    rotation = np.asarray(object_rotation, float)
    if zone.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError('Actual finite zone and object rotation required')
    xyz(zone)
    closing = zone + rotation @ np.array([0., 0., .028]) if case == 'N' else zone.copy()
    return dict(closing=closing, above=closing + [0., 0., .15],
                lift=closing + [0., 0., .08])


def _entry(scene):
    entries = scene.grasp_assist._entries
    item = scene.backend.objects_dict[scene.item_name]
    if len(entries) != 1 or entries[0][0].object_body != item.root_body:
        raise ValueError('P201 requires exactly its knife assist entry')
    return entries[0]


def zone_pose(scene):
    _, body, site, _ = _entry(scene)
    data = scene.grasp_assist.data
    return (np.asarray(data.site_xpos[site], float).copy(),
            np.asarray(data.xmat[body], float).reshape(3, 3).copy())


def eligibility_before(scene):
    """Record the existing expression at the actual pre-control state."""
    assist = scene.grasp_assist
    zone, _, site, _ = _entry(scene)
    left = np.asarray(assist.data.site_xpos[assist._left], float)
    right = np.asarray(assist.data.site_xpos[assist._right], float)
    point = np.asarray(assist.data.site_xpos[site], float)
    center = np.asarray(assist.data.site_xpos[assist._center], float)
    rotation = np.asarray(assist.data.site_xmat[assist._center], float).reshape(3, 3)
    gap_vector = right - left
    gap = float(np.linalg.norm(gap_vector))
    valid_gap = bool(np.isfinite(gap) and gap > 1e-6)
    along = float(np.dot(point - left, gap_vector / gap)) if valid_gap else None
    delta = rotation.T @ (point - center)
    within = np.abs(delta) <= np.asarray(zone.tolerance_xyz)
    between = bool(valid_gap and 0. <= along <= gap)
    eligible = bool(assist._eligible(site, zone.tolerance_xyz))
    return dict(sequence_before=scene.sequence,
        sim_time_s=float(assist.data.time), already_bound=bool(assist.held),
        eligible_branch_used_if_unbound=not bool(assist.held),
        pad_left_world_m=xyz(left), pad_right_world_m=xyz(right),
        zone_world_m=xyz(point), grip_site_world_m=xyz(center),
        pad_gap_m=gap, valid_pad_gap=valid_gap, along_from_left_pad_m=along,
        between_pad_sites=between, gripper_frame_delta_m=xyz(delta),
        tolerance_xyz_m=xyz(zone.tolerance_xyz),
        within_each_axis=dict(zip(('x', 'y', 'z'), map(bool, within))),
        original_eligible=eligible,
        expression_agrees=eligible == bool(between and np.all(within)))


def contact_snapshot(scene, *, combined_aabb, contact_force):
    """Read actual knife/handle contacts and full collision bounds offline."""
    assist = scene.grasp_assist
    model, data = assist.model, assist.data
    item = scene.backend.objects_dict[scene.item_name]
    _, body, site, _ = _entry(scene)
    handles = [name for name in item.contact_geoms
               if name == 'handle' or name.endswith('_handle')]
    if len(handles) != 1:
        raise ValueError('Actual P201 collision handle geom not uniquely identified')
    handle_id = model.geom(handles[0]).id
    contacts, pad_sides, loaded_pad_sides = [], set(), set()
    for index, contact in enumerate(data.contact[:data.ncon]):
        pair = {int(contact.geom1), int(contact.geom2)}
        if contact.dist > 0. or handle_id not in pair or len(pair) != 2:
            continue
        external = next(iter(pair - {handle_id}))
        name = model.geom(external).name or ''
        finger = FINGER.fullmatch(name)
        if finger is None:
            continue
        force = np.zeros(6)
        contact_force(model, data, index, force)
        side = int(finger.group(1))
        is_pad = '_pad_collision' in name
        contacts.append(dict(finger_geom=name, handle_geom=handles[0], side=side,
            pad=is_pad, distance_m=float(contact.dist),
            normal_force_n=float(force[0])))
        if is_pad:
            pad_sides.add(side)
            if force[0] > 1e-6:
                loaded_pad_sides.add(side)
    low, _ = combined_aabb(scene.backend.sim, item.contact_geoms)
    bottom = float(low[2])
    table = float(data.site_xpos[model.site('table_top').id][2])
    return dict(sequence=scene.sequence, sim_time_s=float(data.time),
        assisted_attached=bool(assist.held),
        handle_finger_contacts=contacts, pad_handle_sides=sorted(pad_sides),
        load_bearing_pad_handle_sides=sorted(loaded_pad_sides),
        object_root_world_m=xyz(data.xpos[body]), zone_world_m=xyz(data.site_xpos[site]),
        eef_world_m=xyz(scene.raw['robot0_eef_pos']),
        object_bottom_z_m=bottom, table_top_z_m=table,
        object_clearance_above_table_m=bottom - table)


def run_fixed_case(case, scene, primitives, *, snapshot, check_limits,
                   stage_context, stills, row):
    """The sole fixed trajectory; fake primitives are used in CPU tests."""
    row.update(registration(case), status='running', stages=[])
    row['initial'] = snapshot()
    stills('initial')

    def execute(stage, name, **arguments):
        check_limits()
        stage_context['stage'] = stage
        entry = dict(stage=stage, primitive=name, sequence_before=scene.sequence,
                     arguments={k: xyz(v) if k == 'xyz' else v
                                for k, v in arguments.items()})
        row['stages'].append(entry)
        try:
            result = getattr(primitives, name)(**arguments)
            entry['returned_normally'] = True
            if isinstance(result, dict):
                entry['tool_feedback'] = {k: result[k] for k in
                    ('steps', 'steps_used', 'controls_executed', 'final_dist_m', 'max_steps')
                    if k in result}
        finally:
            entry['sequence_after'] = scene.sequence
        check_limits()
        if name == 'move_to':
            error = float(np.linalg.norm(np.asarray(scene.raw['robot0_eef_pos'])
                                        - np.asarray(arguments['xyz'])))
            entry['actual_eef_error_m'] = error
            if error > MOVE['tol']:
                raise RuntimeError(f'{stage} did not reach fixed 4-mm tolerance; no correction')

    execute('open', 'set_gripper', gripper=-1., steps=12)
    zone, rotation = zone_pose(scene)
    targets = fixed_targets(case, zone, rotation)
    row['fixed_targets_world_m'] = {k: xyz(v) for k, v in targets.items()
                                    if case == 'P' or k != 'lift'}
    row['target_freeze_sequence'] = scene.sequence
    row['pre_move_zone_world_m'] = xyz(zone)
    execute('above', 'move_to', xyz=targets['above'].tolist(), gripper=-1., **MOVE)
    execute('descend', 'move_to', xyz=targets['closing'].tolist(), gripper=-1., **MOVE)
    execute('close', 'set_gripper', gripper=1., steps=12)
    closed = row['closed'] = snapshot()
    stills('closed')
    final = closed
    if case == 'P':
        execute('lift', 'move_to', xyz=targets['lift'].tolist(), gripper=1., **MOVE)
        final = snapshot()
        stills('lifted')
    row['final'] = final
    bottom_lift = final['object_bottom_z_m'] - row['initial']['object_bottom_z_m']
    row['observed_full_bottom_lift_m'] = bottom_lift
    close_samples = [r for r in row.get('control_snapshots', []) if r['stage'] == 'close']
    close_samples.append(dict(closed, stage='close'))
    two_loaded_pads = any(r['load_bearing_pad_handle_sides'] == [1, 2]
                          for r in close_samples)
    actual_pad_contact = any(r['pad_handle_sides'] for r in close_samples)
    unbound_close_eligible = any(r['stage'] == 'close' and not r['already_bound']
        and r['original_eligible'] for r in row.get('before_positive_controls', []))
    bound_no_pad = bool(closed['assisted_attached'] and not closed['pad_handle_sides'])
    row['bind_without_actual_pad_handle_contact'] = bound_no_pad
    row['unbound_close_original_eligible'] = unbound_close_eligible
    row['actual_pad_handle_contact_observed_during_close'] = bool(actual_pad_contact)
    # v4.2 uses the original between-finger/tolerance assist qualification.
    # Two-pad force is a recorded diagnostic, not an additional qualification.
    row['two_load_bearing_pads_observed_during_close'] = two_loaded_pads
    if case == 'N':
        row['case_passed'] = not closed['assisted_attached']
        row['finding'] = ('Negative: no binding at the fixed old high point.'
            if row['case_passed'] else
            'Negative did not pass: binding occurred; inspect original tolerance and actual pad volume/contact relationship without changing tolerance.')
    else:
        row['case_passed'] = bool(closed['assisted_attached'] and
            final['assisted_attached'] and unbound_close_eligible
            and actual_pad_contact and bottom_lift >= .020
            and final['object_clearance_above_table_m'] >= .020)
        row['finding'] = ('Positive fixed contact/lift requirements observed.'
            if row['case_passed'] else
            'Positive requirements not all observed; retain binding, contact and full-bottom measurements. No empty-grasp inference from no assist.')
    row['status'] = 'completed'
    row['physical_controls'] = scene.sequence - row['initial']['sequence']
    return row


def main(case, output, gpu):
    row = registration(case)
    output = Path(output).resolve()
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(gpu),
                      SAFEVLA_SOURCE_CUSTODY='0', SAFEVLA_RUN_OUTPUT=str(output))
    from research.v4.bridge.run_b0 import bind_rpent_checkout
    bind_rpent_checkout()
    import mujoco
    from PIL import Image
    from robots.libero.tools import LiberoPrimitives
    from research.v3.bridge.libero_env import V3LiberoEnvAdapter
    from research.v3.tasks.p2.evaluator import _combined_aabb
    from research.v4.bridge.t1_lite import public_capture
    from research.v4.leash_v43.run_manifest import run_manifest, atomic_json
    from .native import robot_job
    from .p2_contact_zone import Scene, VERSION as ZONE_VERSION, INITIAL_RESET_RECIPE
    from .common_control import CommonControl

    sources = (
        'research/v4/development/confidence_gate_dev_20261006/p2_contact_zone_check.py',
        'research/v4/development/confidence_gate_dev_20261006/p2_contact_zone.py',
        'research/v4/development/confidence_gate_dev_20261006/common_control.py',
        'research/v4/development/common_translation_dev_20261005/common_motion.py',
        'research/v4/development/transport_workspace_dev_20261005/world_reference_controller.py',
        'research/v4/development/transport_workspace_dev_20261005/controller_intent.py',
        'research/v4/candidates/user_rework_p2_20260927/scene_v3.py',
        'research/v4/candidates/user_rework_p2_20260927/scene.py',
        'research/v4/candidates/p2_v42_t3_20260925/scene_safe_slot_v2.py',
        'research/v4/candidates/p2_v42_t3_20260925/knife_visible_blade.xml',
        'research/v4/accelerated/tabletop.py', 'research/v4/accelerated/grasp_assist.py',
        'research/v3/bridge/libero_env.py', 'research/v3/tasks/p2/evaluator.py',
        'research/v4/bridge/t1_lite.py',
        'upstream/RPent-reproduce-libero/robots/libero/tools.py')
    started, scene, manager = time.monotonic(), None, None
    stage = {'stage': 'initial'}
    row.update(contact_zone_version=ZONE_VERSION, initial_reset_recipe=INITIAL_RESET_RECIPE,
               control_snapshots=[], before_positive_controls=[], stills=[])

    def expire(*_):
        raise TimeoutError('180-second fixed P201 contact check limit')

    def size():
        return sum(p.stat().st_size for p in output.rglob('*') if p.is_file())

    def limits():
        if time.monotonic() - started > WALL_S:
            expire()
        if output.exists() and size() > OUTPUT_BYTES:
            raise RuntimeError('100-MiB fixed P201 contact check output limit')

    def snapshot():
        return contact_snapshot(scene, combined_aabb=_combined_aabb,
                                contact_force=mujoco.mj_contactForce)

    def observe(_):
        row['control_snapshots'].append(dict(snapshot(), stage=stage['stage']))
        if scene.sequence > row['max_controls']:
            raise RuntimeError('Fixed predeclared control limit exceeded')
        limits()

    def stills(label):
        limits()
        packet = public_capture(scene)
        images = {view: packet['cameras'][view]['rgb'] for view in
                  ('agentview', 'robot0_eye_in_hand')}
        images['research_spectator'] = np.ascontiguousarray(scene.env.sim.render(
            width=1920, height=1080, camera_name=scene.spectator_view)[::-1])
        for view, pixels in images.items():
            name = f'{case}_{label}_{view}.png'
            Image.fromarray(pixels).save(output / name)
            row['stills'].append(dict(stage=label, view=view, path=name,
                                      sequence=scene.sequence))
            limits()

    previous = signal.signal(signal.SIGALRM, expire)
    try:
        with robot_job(), run_manifest(output, repo=ROOT,
            method='offline_fixed_contact_check', instance_id='P2-01', seed=0,
            gates={}, planner_model=None, planner_reasoning=None,
            reviewer_model=None, reviewer_reasoning=None, source_files=sources,
            run_type='confidence_contact_development', budgets=row['budgets']):
            atomic_json(output / 'results.json', dict(rows=[row], models=0,
                physical_controls=0, offline_only=True, empty_fact_issued=False),
                max_output_bytes=OUTPUT_BYTES, budget_directory=output)
            signal.setitimer(signal.ITIMER_REAL,
                max(.001, WALL_S - (time.monotonic() - started)))
            try:
                scene = Scene(0, camera_observations=True, render_gpu_device_id=gpu)
                if scene.sequence != 0 or scene.initial_reset_recipe != INITIAL_RESET_RECIPE:
                    raise ValueError('A fresh corrected P201 constructor is required')
                entry = _entry(scene)
                actual_zone = scene.grasp_assist.model.site_pos[entry[2]]
                if tuple(map(float, actual_zone)) != (-.045, 0., 0.):
                    raise ValueError('Actual compiled contact zone differs from the fixed overlay')
                row['compiled_zone_local_m'] = xyz(actual_zone)
                row['unchanged_tolerance_xyz_m'] = xyz(entry[0].tolerance_xyz)
                adapter = V3LiberoEnvAdapter(scene, instruction=scene.instruction,
                                             on_sim_step=observe)
                original_step = adapter.step

                def step(_, action):
                    limits()
                    if float(action[6]) > 0.:
                        row['before_positive_controls'].append(dict(
                            eligibility_before(scene), stage=stage['stage']))
                    return original_step(action)

                adapter.step = MethodType(step, adapter)
                primitives = LiberoPrimitives(env=adapter, model=None,
                    sam3_client=None, check_cancelled=limits)
                primitives.reset()
                manager = CommonControl(primitives)
                manager.install()
                run_fixed_case(case, scene, primitives, snapshot=snapshot,
                    check_limits=limits, stage_context=stage, stills=stills, row=row)
            except BaseException as exc:
                row.update(status='failed', case_passed=False,
                           error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                try:
                    try:
                        if manager is not None:
                            manager.close()
                    finally:
                        if scene is not None:
                            row['physical_controls'] = scene.sequence
                            scene.close()
                finally:
                    atomic_json(output / 'results.json', dict(rows=[row], models=0,
                        physical_controls=row.get('physical_controls', 0),
                        offline_only=True, empty_fact_issued=False,
                        online_dynamic_truth=False), max_output_bytes=OUTPUT_BYTES,
                        budget_directory=output)
                    atomic_json(output / 'accounting.json', dict(
                        wall_s=time.monotonic()-started,
                        utc=datetime.now(timezone.utc).isoformat()),
                        max_output_bytes=OUTPUT_BYTES, budget_directory=output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    print(json.dumps(dict(case=case, case_passed=row['case_passed'],
        physical_controls=row['physical_controls'], wall_s=time.monotonic()-started,
        output_bytes=size(), models=0)))
    return 0 if row['case_passed'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=('N', 'P'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int, choices=(0, 1))
    args = parser.parse_args()
    raise SystemExit(main(args.case, args.output, args.gpu))
