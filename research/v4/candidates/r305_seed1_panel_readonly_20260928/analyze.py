"""Saved R305 trace/model arithmetic only; no Scene, IK, forward or controls."""
from pathlib import Path
import hashlib
import itertools
import json
import time

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OWN = HERE.parent / 'r305_tall_microwave_inside12_20260928'
FAIL = HERE.parent / 'r305_tall350_root_qualification_20260928/seeds1_9/naive_seed1.json'
PASS = OWN / 'root_seed0_scripts/naive_seed0.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vertices(m, d, gid):
    R = d.geom_xmat[gid].reshape(3, 3)
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[gid]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        local = m.mesh_vert[start:start + count].astype(float)
    elif m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.array(list(itertools.product((-1., 1.), repeat=3))) * m.geom_size[gid]
    else:
        raise RuntimeError('Uninterpreted hard geometry type: ' + str(m.geom(gid).name))
    return local @ R.T + d.geom_xpos[gid]


def bbox(v):
    return [v.min(axis=0).tolist(), v.max(axis=0).tolist()]


def sat_mesh_box(mesh, box, box_axes):
    """Convex collision mesh vs actual box, signed separating-axis gap, meters."""
    hull = ConvexHull(mesh)
    edges = np.concatenate([mesh[hull.simplices[:, (j + 1) % 3]] - mesh[hull.simplices[:, j]]
                            for j in range(3)])
    axes = np.concatenate([hull.equations[:, :3], box_axes.T,
                           np.cross(edges[:, None, :], box_axes.T[None, :, :]).reshape(-1, 3)])
    norm = np.linalg.norm(axes, axis=1)
    axes = axes[norm > 1e-12] / norm[norm > 1e-12, None]
    a, b = mesh @ axes.T, box @ axes.T
    gap = np.maximum(b.min(axis=0) - a.max(axis=0), a.min(axis=0) - b.max(axis=0))
    i = int(np.argmax(gap))
    return dict(max_separating_gap_m=float(gap[i]), axis=axes[i].tolist(),
                convex_hulls_intersect=bool(gap[i] < 0),
                scope='Static compiled convex collision geometry, not robot reach or dynamic force')


def main():
    start = time.monotonic()
    out = HERE / 'report.json'
    if out.exists():
        raise RuntimeError('Readonly receipt is write-once')
    runs = [json.loads(PASS.read_text()), json.loads(FAIL.read_text())]
    m = mujoco.MjModel.from_binary_path(str(OWN / 'actual_constructor_model.mjb'))
    center = m.site('gripper0_right_grip_site').id
    panel, wrist = m.geom('microwave_1_g8').id, m.geom('robot0_link7_collision').id
    result = dict(Scene_constructors=0, model_compiles=0, actual_controls=0, IK_calls=0,
                  mj_forward_calls=0, GPU_models_services=0, seed1_model_saved=False,
                  scope='Load existing seed0 MJB; independent saved-qpos kinematics and rigid mesh arithmetic only',
                  whole_seed1_model_identity_claim=False, contact_force_saved=False,
                  panel=dict(name=m.geom(panel).name, body=m.body(m.geom_bodyid[panel]).name,
                             type=int(m.geom_type[panel]), pos=m.geom_pos[panel].tolist(),
                             size=m.geom_size[panel].tolist(), contype=int(m.geom_contype[panel]),
                             conaffinity=int(m.geom_conaffinity[panel]), margin=float(m.geom_margin[panel]),
                             interpretation='Real solid front control-panel housing block, not roof g6'),
                  seeds=[])
    for run in runs:
        data = mujoco.MjData(m)
        contacts = [(i + 1, row, c) for i, row in enumerate(run['actual_control_rows'])
                    for c in row['contacts'] if set(c['geoms']) ==
                    {'microwave_1_g8', 'robot0_link7_collision'}]
        item = dict(seed=run['seed'], passed=run['passed'], error=run['error'],
                    controls=run['control_steps'], total_controls=run['total_controls_including_common_reset'],
                    stages=run['stages'], panel_wrist_contact_rows=len(contacts),
                    samples=[], contact_dist_range_m=(
                        [min(c['dist_m'] for _, _, c in contacts), max(c['dist_m'] for _, _, c in contacts)]
                        if contacts else None))
        if contacts:
            i, row, c = contacts[0]
            item['first_contact'] = dict(episode_control=i, sequence=row['sequence'],
                                         time=row['time'], EEF=row['EEF'], contact=c)
            item['last_contact'] = dict(episode_control=contacts[-1][0], contact=contacts[-1][2])
        for i in (1, 60, 67, 68, 69, 100, min(180, len(run['actual_control_rows']))):
            row = run['actual_control_rows'][i - 1]
            data.qpos[:] = row['qpos']
            mujoco.mj_kinematics(m, data)
            item['samples'].append(dict(episode_control=i, EEF=row['EEF'],
                saved_EEF_vs_savedqpos_FK_max_delta=float(np.max(np.abs(data.site_xpos[center] - row['EEF']))),
                action=row['action'], q7=row['qpos'][6], qvel7=row['qvel'][6],
                robot_joint_margin_rad=np.minimum(np.array(row['qpos'][:7]) - m.jnt_range[:7, 0],
                                                  m.jnt_range[:7, 1] - row['qpos'][:7]).tolist(),
                panel_bbox=bbox(vertices(m, data, panel)), wrist_bbox=bbox(vertices(m, data, wrist))))
        # Existing desired stage endpoint only; this is not a newly selected route/IK pose.
        row = run['actual_control_rows'][min(180, len(run['actual_control_rows'])) - 1]
        data.qpos[:] = row['qpos']; mujoco.mj_kinematics(m, data)
        goal = np.asarray(run['stages'][0]['goal_m'])
        actual_p, actual_R = data.site_xpos[center], data.site_xmat[center].reshape(3, 3)
        desired_R = np.asarray(run['actual_private_lug_design']['target_R'])
        panel_v = vertices(m, data, panel)
        wrist_v = vertices(m, data, wrist)
        placed = ((wrist_v - actual_p) @ actual_R) @ desired_R.T + goal
        item['original_above_target_rigid_wrist'] = dict(EEF=goal.tolist(), wrist_bbox=bbox(placed),
            panel_bbox=bbox(panel_v), separation=sat_mesh_box(placed, panel_v, data.geom_xmat[panel].reshape(3, 3)))
        initial = run['actual_control_rows'][0]
        item['initial_existing_EEF_height_m'] = initial['EEF'][2]
        result['seeds'].append(item)
    result['elapsed_s'] = time.monotonic() - start
    result['source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in
                              (Path(__file__), FAIL, PASS, OWN / 'actual_constructor_model.mjb',
                               OWN / 'actual_constructor_model.xml', OWN / 'expert.py', OWN / 'scene.py')}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), seeds=[dict(seed=r['seed'], contacts=r['panel_wrist_contact_rows'],
        static_above_target=r['original_above_target_rigid_wrist']['separation'])for r in result['seeds']]), indent=2))


if __name__ == '__main__':
    main()
