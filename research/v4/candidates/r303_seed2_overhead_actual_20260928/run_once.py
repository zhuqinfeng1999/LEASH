"""Root-only one R3-03 seed-2 naive script with an overhead free-hand approach."""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from research.v4.candidates.r303_base_forward90_actual_seed4_20260928 import run_once as selected_physical
from research.v4.candidates.r3_03_own_horizontal_actual_scripts_20260928 import expert

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
COPIED = ROOT / 'research/v4/candidates/r303_seed2_overhead_exactpre_20260928/EXACT_PRE_RECEIPT.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def z_support(m, d, g, eef, eef_R, target_R):
    p = eef_R.T @ (d.geom_xpos[g] - eef)
    Q = eef_R.T @ d.geom_xmat[g].reshape(3, 3)
    direction = target_R.T @ np.array([0., 0., 1.])
    centre = float(direction @ p)
    u = Q.T @ direction
    typ = int(m.geom_type[g])
    if typ == int(mujoco.mjtGeom.mjGEOM_BOX):
        extent = float(np.abs(u) @ m.geom_size[g])
        return centre - extent, centre + extent
    if typ == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius, halfheight = m.geom_size[g, :2]
        extent = float(radius * np.hypot(u[0], u[1]) + halfheight * abs(u[2]))
        return centre - extent, centre + extent
    if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        extent = float(m.geom_size[g, 0]); return centre - extent, centre + extent
    if typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, halfheight = m.geom_size[g, :2]
        extent = float(radius + halfheight * abs(u[2]))
        return centre - extent, centre + extent
    if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(m.geom_dataid[g]); start = int(m.mesh_vertadr[mesh]); n = int(m.mesh_vertnum[mesh])
        verts = p + (Q @ m.mesh_vert[start:start+n].T).T
        values = verts @ direction
        return float(values.min()), float(values.max())
    raise ValueError(f'Unsupported actual geom type {typ}: {m.geom(g).name}')


def overhead_height(scene, source_R):
    m, d = scene.backend.sim.model._model, scene.backend.sim.data._data
    center = scene.grasp_assist._center
    home = d.site_xpos[center].copy()
    home_R = d.site_xmat[center].reshape(3, 3).copy()
    thermos = [g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('steel_thermos_1_')]
    hand = [g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('gripper0_')
            and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if not thermos or not hand:
        raise ValueError('Compiled thermos/hand solids missing')
    source_top = max(z_support(m, d, g, np.zeros(3), np.eye(3), np.eye(3))[1] for g in thermos)
    hand_low_rel = min(z_support(m, d, g, home, home_R, source_R)[0] for g in hand)
    height = max(float(home[2]), float(source_top + .010 - hand_low_rel))
    copied = json.loads(COPIED.read_text())
    expected = copied.get('exact_pregrasp_EEF_error_m')
    if not copied['passed'] or expected is None or expected > .0005:
        raise ValueError('Same-route copied overhead check not qualified')
    return height, {'source_top_world_z_m': source_top,
                    'full_hand_low_relative_z_m': hand_low_rel,
                    'clearance_m': .010, 'derived_EEF_world_z_m': height,
                    'thermos_geom_count': len(thermos), 'hand_geom_count': len(hand)}


def selected_naive():
    src = inspect.getsource(expert.run_naive)
    before = "    pose(pre,sourceR,'home_to_horizontal_pre',tol=.0005,rotation_tol=np.deg2rad(.5))"
    after = """    home=d.site_xpos[center].copy();homeR=d.site_xmat[center].reshape(3,3).copy()
    high,clearance=overhead_height(scene,sourceR)
    details['single_overhead_approach_geometry']=clearance
    raised=home.copy();raised[2]=high
    above=pre.copy();above[2]=high
    pose(raised,homeR,'raise_home_clearance')
    pose(raised,sourceR,'orient_at_clearance')
    pose(above,sourceR,'lateral_above_original_pregrasp')
    pose(pre,sourceR,'descend_at_original_pregrasp_XY',clip=.006,tol=.0005,rotation_tol=np.deg2rad(.5))"""
    if src.count(before) != 1:
        raise ValueError('Original naive source-approach seam changed')
    src = src.replace(before, after)
    namespace = dict(expert.run_naive.__globals__, overhead_height=overhead_height)
    exec(compile(src, str(H / 'run_once.py')+'::one-overhead-approach', 'exec'), namespace)
    return namespace['run_naive'], src


def admission():
    pinned = json.loads((H / 'source_freeze.json').read_text())
    for name, digest in pinned.items():
        path = Path(name)
        if not path.is_absolute(): path = ROOT / path
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Overhead candidate source drift: '+name)
    selected_physical.hashes()
    original, case_source, _ = selected_physical.selected()
    new_naive, naive_source = selected_naive()
    if original.__globals__['run_naive'] is not expert.run_naive:
        raise ValueError('Original generated run_case naive seam changed')
    if original.__globals__['R3TabletopFacade'] is not selected_physical.original.SelectedOwnThermosScene:
        raise ValueError('Original physical Scene seam changed')
    if original.__globals__['MAX_STEPS'] != 1200 or original.__globals__['MAX_WALL_S'] != 90.0:
        raise ValueError('Original budget changed')
    return pinned, original, case_source, new_naive, naive_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    pinned, f, original_source, naive, naive_source = admission()
    summary = {'instance': 'R3-03', 'seed': 2, 'mode': 'naive', 'only_one_actual_trial': True,
               'new_physical_version': 'forward90-same-thermos-oven',
               'pinned_sources': len(pinned), 'original_generated_case_sha256': hashlib.sha256(original_source.encode()).hexdigest(),
               'one_naive_approach_source_sha256': hashlib.sha256(naive_source.encode()).hexdigest(),
               'original_caps': [1200, 90.0], 'method_result': False}
    if args.preflight_only:
        print(json.dumps(dict(summary, task_controls=0, Scene_constructors=0, GPU=False)))
        return
    if not args.output: raise ValueError('Fresh output required')
    out = Path(args.output).resolve()
    if out.exists(): raise ValueError('Never overwrite/retry the preserved seed2 failure')
    out.mkdir(parents=True)
    (out / 'selected_naive_approach_source.py.txt').write_text(naive_source)
    (out / 'selected_original_case_source.py.txt').write_text(original_source)
    started = time.monotonic()
    try:
        with patch.dict(f.__globals__, {'R3TabletopFacade': selected_physical.make_scene,
                                        'run_naive': naive}):
            if f.__globals__['run_naive'] is not naive or f.__globals__['R3TabletopFacade'] is not selected_physical.make_scene:
                raise RuntimeError('Selected original case did not consume the isolated bindings')
            result = f(2, 'naive')
        (out / 'naive_seed2.json').write_text(json.dumps(result, indent=2)+'\n')
        summary['outcome'] = {'passed': result['passed'], 'label': result['evaluator']['label'],
                              'control_steps': result['control_steps'], 'error': result['error'],
                              'true_bind': result.get('true_bind'), 'held_after_lift': result.get('held_after_lift'),
                              'lift_m': result.get('lift_m')}
    except BaseException as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        summary['elapsed_s'] = time.monotonic()-started
        summary['source_drift'] = [name for name, digest in pinned.items()
                                   if sha(Path(name) if Path(name).is_absolute() else ROOT/name) != digest]
        (out / 'receipt.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:summary.get(k) for k in ('outcome','error','elapsed_s','source_drift')}))
    if summary.get('error') or summary['source_drift'] or not summary.get('outcome',{}).get('passed'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
