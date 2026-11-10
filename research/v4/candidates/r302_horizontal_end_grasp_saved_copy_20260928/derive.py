"""One exact own-can/compiled-hand material geometry derivation; no Scene."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
SAVED = ROOT / 'research/v4/candidates/r302_rear_presentation_v1_20260928/support_seed0_v2'
PAD_NAMES = ('gripper0_right_finger1_pad_collision', 'gripper0_right_finger2_pad_collision')
CAN_NAMES = ('steel_thermos_1_can_body_collision',
             'steel_thermos_1_can_rim_collision_-0.05575',
             'steel_thermos_1_can_rim_collision_0.05575')


def load():
    m = mujoco.MjModel.from_binary_path(str(SAVED / 'compiled_model.mjb'))
    data = np.load(SAVED / 'constructor_data.npz', allow_pickle=False)
    d = mujoco.MjData(m)
    d.qpos[:] = data['qpos']; d.qvel[:] = data['qvel']
    d.ctrl[:] = data['ctrl']; d.time = float(data['time'])
    mujoco.mj_forward(m, d)
    return m, d


def hand_geoms(m):
    return [i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('gripper0_right_')
            and (m.geom_contype[i] or m.geom_conaffinity[i])]


def hand_geometry(m, d):
    sid = m.site('gripper0_right_grip_site').id
    p = d.site_xpos[sid].copy(); R = d.site_xmat[sid].reshape(3, 3).copy()
    rows = []
    for i in hand_geoms(m):
        gr = d.geom_xmat[i].reshape(3, 3)
        row = {'name': m.geom(i).name, 'id': i, 'type': int(m.geom_type[i]),
               'local_position': R.T @ (d.geom_xpos[i] - p), 'local_rotation': R.T @ gr,
               'size': m.geom_size[i].copy()}
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            k = int(m.geom_dataid[i]); start = int(m.mesh_vertadr[k]); n = int(m.mesh_vertnum[k])
            vertices = m.mesh_vert[start:start+n].astype(float)
            row['vertices_eef'] = (vertices @ gr.T + d.geom_xpos[i] - p) @ R
        rows.append(row)
    return rows


def desired_frame(m, d):
    bid = m.body('steel_thermos_1_main').id
    B = d.xmat[bid].reshape(3, 3).copy()
    R = np.column_stack((B[:, 1], B[:, 0], -B[:, 2]))
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-12) or np.linalg.det(R) < .999999:
        raise RuntimeError('Own material-axis frame not a proper rotation')
    z = d.site_xpos[m.site('v42_steel_thermos_1_grasp_zone').id].copy()
    return z, R


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [serial(v) for v in x]
    return x


def main():
    out = H / 'geometry.json'
    if out.exists(): raise RuntimeError('Preserve first own geometry derivation')
    m, d = load()
    fadr = [int(m.jnt_qposadr[m.joint(n).id]) for n in (
        'gripper0_right_finger_joint1', 'gripper0_right_finger_joint2')]
    p, R = desired_frame(m, d)
    can_b = m.body('steel_thermos_1_main').id
    if abs(float(m.body_mass[can_b]) - .4) > 1e-12:
        raise RuntimeError('Own can mass changed')
    barrel = m.geom(CAN_NAMES[0]).id
    if int(m.geom_type[barrel]) != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        raise RuntimeError('Own cylinder type changed')
    # pad local center is q+3.5 mm; half-width in closing direction is 4 mm.
    q_close = float(m.geom_size[barrel, 0]) + .0005
    env = [i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('r302_v_')
           or m.geom(i).name == 'table_collision']
    if len(env) != 4: raise RuntimeError('Expected both rails, base and exact table')
    cans = [m.geom(n).id for n in CAN_NAMES]
    result = {'kind':'own saved-model horizontal material geometry', 'Scene_constructors':0,
              'physics_steps':0, 'task_controls':0, 'passed':False, 'failure':None,
              'can_mass_kg':float(m.body_mass[can_b]), 'zone_local_m':m.site_pos[m.site('v42_steel_thermos_1_grasp_zone').id],
              'source_zone_m':p, 'source_R_eef':R, 'source_can_R':d.xmat[can_b].reshape(3,3).copy(),
              'approach_mouth_dot':float(np.dot(R[:,2],d.site_xmat[m.site('microwave_1_heating_region').id].reshape(3,3)[:,1])),
              'q_close_m':q_close,'original_tolerance_m':[.015,.015,.020], 'pairs':[], 'pad_metadata':[]}
    for name in PAD_NAMES:
        i=m.geom(name).id
        row={'name':name,'type':int(m.geom_type[i]),'size_m':m.geom_size[i].copy(),
             'solref':m.geom_solref[i].copy(),'masks':[int(m.geom_contype[i]),int(m.geom_conaffinity[i])]}
        if row['type'] != 6 or not np.array_equal(m.geom_size[i],[.008,.004,.008]) or not np.array_equal(m.geom_solref[i],[.01,.5]):
            raise RuntimeError('Exact own soft-pad metadata mismatch')
        result['pad_metadata'].append(row)
    geometry={}
    for stage, q in [('open',.04),('material_closed',q_close)]:
        d.qpos[fadr]=[q,-q];mujoco.mj_forward(m,d)
        geoms=hand_geometry(m,d);geometry[stage]=geoms
        # These rigid shape transforms are a pure geometric query. They do not
        # set robot qpos, a weld, an object pose or any physical control.
        for row in geoms:
            i=row['id'];d.geom_xpos[i]=p+R@row['local_position'];d.geom_xmat[i]=(R@row['local_rotation']).reshape(9)
        for row in geoms:
            i=row['id']
            for j in cans+env:
                line=np.zeros(6)
                dist=float(mujoco.mj_geomDistance(m,d,i,j,.25,line))
                pair={'stage':stage,'geoms':[m.geom(i).name,m.geom(j).name],
                      'signed_distance_m':dist,'nearest_segment_world_m':line,
                      'exact_pad_can':row['name'] in PAD_NAMES and j in cans}
                result['pairs'].append(pair)
    nonpads=[r for r in result['pairs'] if not r['exact_pad_can']]
    bad=[r for r in nonpads if r['signed_distance_m'] < -.001]
    result['min_nonpad_signed_distance_m']=min(r['signed_distance_m'] for r in nonpads)
    result['closed_barrel_pad_distances_m']=[r['signed_distance_m'] for r in result['pairs']
        if r['stage']=='material_closed' and r['geoms'][0] in PAD_NAMES and r['geoms'][1]==CAN_NAMES[0]]
    result['hard_finger_can_pairs']=[r for r in result['pairs'] if '_finger' in r['geoms'][0]
        and '_pad_' not in r['geoms'][0] and r['geoms'][1] in CAN_NAMES]
    if bad: result['failure']={'kind':'first non-pad material/fixture intersection deeper than 1mm','pair':bad[0]}
    elif any(abs(x)>.0001 for x in result['closed_barrel_pad_distances_m']):
        result['failure']={'kind':'derived closed pad faces do not contact own real barrel'}
    else: result['passed']=True
    result['input_sha256']={str((SAVED/n).relative_to(ROOT)):hashlib.sha256((SAVED/n).read_bytes()).hexdigest()
                           for n in ('compiled_model.mjb','constructor_data.npz','report.json')}
    (H/'hand_geometry.json').write_text(json.dumps(serial(geometry),indent=2)+'\n')
    out.write_text(json.dumps(serial(result),indent=2)+'\n')
    print(json.dumps(serial({k:result[k] for k in ('passed','failure','q_close_m','min_nonpad_signed_distance_m','closed_barrel_pad_distances_m')})))


if __name__=='__main__':main()
