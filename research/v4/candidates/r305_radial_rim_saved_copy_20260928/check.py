"""One radial ceramic-rim grasp on the saved own model; no new Scene or controls."""
import hashlib
import itertools
import json
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.accelerated.slot_preflight import _ik

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
SAVED = H.parent / 'r305_raised_neutral_copy_20260928'
PADS = ['gripper0_right_finger1_pad_collision', 'gripper0_right_finger2_pad_collision']


def contact_rows(m, d):
    rows = []
    for i, c in enumerate(d.contact[:d.ncon]):
        names = [m.geom(int(g)).name or '' for g in (c.geom1, c.geom2)]
        if not any(n.startswith(('robot0_', 'gripper0_', 'fork_bowl_1_')) for n in names): continue
        pad = any(n in PADS for n in names) and any(n.startswith('fork_bowl_1_ceramic_') and n.endswith('_collision') for n in names)
        robot = any(n.startswith(('robot0_', 'gripper0_')) for n in names)
        rows.append({'geoms': names, 'dist_m': float(c.dist), 'position': c.pos.tolist(),
                     'exact_pad_material': pad, 'hard_stop': bool(robot and c.dist < -.001 and not pad)})
    return rows


def main():
    if (H / 'report.json').exists(): raise RuntimeError('One first saved-model attempt')
    pins = json.loads((H / 'source_freeze.json').read_text())
    for name, h in pins.items():
        p = Path(name) if Path(name).is_absolute() else ROOT / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != h: raise RuntimeError('Source drift ' + name)
    result = {'passed': False, 'failure': None, 'Scene_constructors': 0, 'task_controls': 0,
              'GPU_models_services': 0, 'source_sha256': pins, 'static_frames': [], 'route': [],
              'not_actual_grasp_or_task_qualification': True}
    start = time.monotonic()
    try:
        m = mujoco.MjModel.from_binary_path(str(SAVED / 'actual_model.mjb'))
        d = mujoco.MjData(m); saved = np.load(SAVED / 'after_neutral_data.npz')
        d.qpos[:] = saved['qpos']; d.qvel[:] = saved['qvel']; mujoco.mj_forward(m, d)
        initial = d.qpos.copy()
        center = m.site('gripper0_right_grip_site').id
        left = m.site('v42_left_pad').id; right = m.site('v42_right_pad').id
        zone = m.site('v42_fork_bowl_1_grasp_zone').id
        body = m.body('fork_bowl_1_main').id; oven = m.body('microwave_1_main').id
        hs = m.site('microwave_1_heating_region').id
        qids = [m.joint(f'robot0_joint{i}').id for i in range(1,8)]
        qidx = [m.jnt_qposadr[j] for j in qids]
        finger = [m.joint('gripper0_right_finger_joint1').id, m.joint('gripper0_right_finger_joint2').id]
        for name in PADS:
            g = m.geom(name).id
            if int(m.geom_type[g]) != 6 or m.geom_size[g].tolist() != [.008,.004,.008] or m.geom_solref[g].tolist() != [.01,.5]:
                raise RuntimeError('Exact compiled soft pad changed')
        if m.site_pos[zone].tolist() != [.059,0.,.022] or float(m.body_mass[body]) != .31:
            raise RuntimeError('Original59mm zone/310g shape source changed')
        p0 = d.site_xpos[center].copy(); Q0 = d.site_xmat[center].reshape(3,3).copy()
        lp = Q0.T @ (d.site_xpos[left]-p0); rp = Q0.T @ (d.site_xpos[right]-p0)
        jaw_axis = (rp-lp)/np.linalg.norm(rp-lp)
        if not np.allclose(jaw_axis,[1.,0.,0.],rtol=0,atol=1e-10): raise RuntimeError('Actual jaw axis differs from derived EEF X')
        radial = d.site_xpos[zone]-d.xpos[body];radial[2]=0.;radial/=np.linalg.norm(radial)
        down = np.array([0.,0.,-1.]); R = np.column_stack((radial,np.cross(down,radial),down))
        B = d.xmat[body].reshape(3,3).copy(); bp = d.xpos[body].copy()
        rims = [g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('fork_bowl_1_ceramic_rim_') and (m.geom(g).name or '').endswith('_collision')]
        if not rims or any(int(m.geom_type[g]) != 3 for g in rims): raise RuntimeError('Original hollow capsule rim changed')
        height = float(np.median([m.geom_pos[g,2] for g in rims]))
        axis_origin = bp + B @ np.array([0.,0.,height])
        far_distance = 2*max(float(np.linalg.norm(m.geom_pos[g,:2]))+float(m.geom_size[g,0]) for g in rims)

        def ray(point, direction):
            hits = []
            for g in rims:
                distance = mujoco.mju_rayGeom(d.geom_xpos[g].copy(),d.geom_xmat[g].copy(),m.geom_size[g].copy(),point,direction,int(m.geom_type[g]))
                if distance >= 0.:hits.append({'geom':m.geom(g).name,'distance':float(distance)})
            if not hits:raise RuntimeError('Actual radial line has no material-rim intersection')
            return min(hits,key=lambda x:x['distance'])

        inner_hit = ray(axis_origin,radial); outer_hit = ray(axis_origin+far_distance*radial,-radial)
        inner = inner_hit['distance'];outer = far_distance-outer_hit['distance']
        material = axis_origin + (inner+outer)/2*radial
        midpoint_local = (lp+rp)/2
        target = material-R@midpoint_local
        half = float(np.sum(np.abs((Q0.T@d.geom_xmat[m.geom(PADS[0]).id].reshape(3,3))[0])*m.geom_size[m.geom(PADS[0]).id]))
        zero_offset_left = abs(float(lp[0]))-float(d.qpos[m.jnt_qposadr[finger[0]]])
        zero_offset_right = abs(float(rp[0]))-abs(float(d.qpos[m.jnt_qposadr[finger[1]]]))
        closed_q = (outer-inner)/2+half-(zero_offset_left+zero_offset_right)/2
        if not 0.<closed_q<=.04:raise RuntimeError('Material/pad-derived width not a legal normal pinch')
        result['design'] = {'jaw_local_axis':jaw_axis.tolist(),'radial_world':radial.tolist(),'approach_world':down.tolist(),
            'actual_zone_world':d.site_xpos[zone].tolist(),'original_zone_local':m.site_pos[zone].tolist(),
            'rim_centerline_height_local':height,'inner_hit':inner_hit,'outer_hit':outer_hit,
            'material_radial_interval':[inner,outer],'material_width_m':outer-inner,
            'actual_pad_half_width_along_jaw':half,'pad_center_offsets_at_zero':[zero_offset_left,zero_offset_right],
            'closed_finger_qpos':[closed_q,-closed_q],'material_center':material.tolist(),
            'actual_pad_midpoint_EEF_local':midpoint_local.tolist(),'target_EEF':target.tolist(),'target_R':R.tolist(),
            'no_home_orientation_reuse':True,'no_force_q0':True}
        proxy = SimpleNamespace(model=SimpleNamespace(_model=m),data=SimpleNamespace(qpos=d.qpos))
        ik = _ik(proxy,target,R,max_iter=200,position_tol=.001,rotation_tol=.03)
        if not ik['reachable'] or ik.get('restart',0)!=0:raise RuntimeError('First fixed radial endpoint IK unavailable continuously')
        d.qpos[qidx]=ik['joint_qpos_rad'];mujoco.mj_forward(m,d)
        for kind, qs in [('actual_open',[float(initial[m.jnt_qposadr[j]]) for j in finger]),('material_width_closed',[closed_q,-closed_q])]:
            for j,q in zip(finger,qs):d.qpos[m.jnt_qposadr[j]]=q
            mujoco.mj_forward(m,d)
            cs = contact_rows(m,d)
            helper=GraspAssist.__new__(GraspAssist);helper.model=m;helper.data=d;helper._center=center;helper._left=left;helper._right=right
            eligible = helper._eligible(zone,(.015,.015,.020))
            frame={'kind':kind,'qpos':d.qpos.tolist(),'actual_EEF':d.site_xpos[center].tolist(),
                   'actual_R':d.site_xmat[center].reshape(3,3).tolist(),'pad_centers':[d.site_xpos[left].tolist(),d.site_xpos[right].tolist()],
                   'finger_qpos':qs,'actual_original_zone_eligible':bool(eligible),'contacts':cs,'ik':ik}
            result['static_frames'].append(frame)
            if any(c['hard_stop']for c in cs):raise RuntimeError('First fixed radial '+kind+' solid hard contact >1mm')
            if kind=='material_width_closed'and not eligible:raise RuntimeError('Material-width closed grasp not eligible under15/15/20')
        result['static_two_frames_passed']=True
        # The sole saved-copy route now starts again from the original saved
        # home; this is independent data, never a live state restore.
        d.qpos[:]=initial;mujoco.mj_forward(m,d);rel=None
        item_joint=m.joint('fork_bowl_1_joint0').id;adr=int(m.jnt_qposadr[item_joint])

        def pose(phase,p,Q):
            ik=_ik(proxy,p,Q,max_iter=200,position_tol=.008,rotation_tol=.03)
            row={'phase':phase,'goal':p.tolist(),'R':Q.tolist(),'ik':ik}
            if not ik['reachable']or ik.get('restart',0)!=0:
                result['route'].append(row);raise RuntimeError('First continuous radial route IK gap '+phase)
            d.qpos[qidx]=ik['joint_qpos_rad'];mujoco.mj_forward(m,d)
            if rel is not None:
                A=d.site_xmat[center].reshape(3,3)
                d.qpos[adr:adr+3]=d.site_xpos[center]+A@rel[0]
                d.qpos[adr+3:adr+7]=Rotation.from_matrix(A@rel[1]).as_quat()[[3,0,1,2]];mujoco.mj_forward(m,d)
            cs=contact_rows(m,d);local=d.site_xmat[hs].reshape(3,3).T@(d.xpos[body]-d.site_xpos[hs])
            row.update(qpos=d.qpos.tolist(),EEF=d.site_xpos[center].tolist(),payload=d.xpos[body].tolist(),contacts=cs,
                       original_critical=bool(np.all(np.abs(local)<m.site_size[hs])))
            result['route'].append(row)
            if any(c['hard_stop']for c in cs):raise RuntimeError('First continuous radial route hard contact >1mm '+phase)

        def segment(phase,p,Q=R):
            old=d.site_xpos[center].copy();oldQ=d.site_xmat[center].reshape(3,3).copy()
            from scipy.spatial.transform import Slerp
            turn=Slerp([0,1],Rotation.from_matrix([oldQ,Q]))
            for t in np.linspace(0,1,max(10,int(np.ceil(np.linalg.norm(p-old)/.025)))+1):
                pose(phase,old*(1-t)+p*t,turn(t).as_matrix())
                if rel is not None and result['route'][-1]['original_critical']:return

        above=target.copy();above[2]=1.15
        segment('radial_above_original_bowl',above)
        segment('radial_vertical_material_approach',target)
        for j,q in zip(finger,[closed_q,-closed_q]):d.qpos[m.jnt_qposadr[j]]=q
        mujoco.mj_forward(m,d)
        helper.data=d
        if not helper._eligible(zone,(.015,.015,.020)):raise RuntimeError('Continuous material-frame closure not eligible')
        cs=contact_rows(m,d)
        result['route_closure']={'qpos':d.qpos.tolist(),'contacts':cs,'eligible':True}
        if any(c['hard_stop']for c in cs):raise RuntimeError('Continuous material-width closed hard contact >1mm')
        A=d.site_xmat[center].reshape(3,3).copy();grip=d.site_xpos[center].copy()
        rel=(A.T@(d.xpos[body]-grip),A.T@d.xmat[body].reshape(3,3))
        oR=d.xmat[oven].reshape(3,3).copy();op=d.xpos[oven].copy()

        def verts(g):
            a=m.geom_aabb[g];G=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g]+G@a[:3]
            return np.array([c+G@(a[3:]*v)for v in itertools.product((-1,1),repeat=3)])

        solid=[g for g in range(m.ngeom)if(m.geom_contype[g]or m.geom_conaffinity[g])and(m.geom(g).name or '').startswith(('fork_bowl_1_','gripper0_'))]
        offset=(np.concatenate([verts(g)for g in solid])-grip)@oR
        floor=((verts(m.geom('microwave_1_g7').id)-op)@oR)[:,2].max()
        roof=((verts(m.geom('microwave_1_g6').id)-op)@oR)[:,2].min()
        lower=float(floor+.005-offset[:,2].min());upper=float(roof-.005-offset[:,2].max())
        carry=max(float(op[2]+lower),float(grip[2]+.020))
        result['derived_lowcarry']={'ovenlocal_interval':[lower,upper],'EEF_world_z':carry,'actual_lift_rule_m':.020,'whole_payload_hand':True}
        if carry>op[2]+upper:raise RuntimeError('Full hand/payload original lowcarry interval empty')
        lift=grip.copy();lift[2]=carry;segment('radial_source_low_lift',lift)
        leftface=((verts(m.geom('microwave_1_g3').id)-op)@oR)[:,0].max()
        rightface=((verts(m.geom('microwave_1_g4').id)-op)@oR)[:,0].min()
        interval=[float(leftface+.005-offset[:,0].min()),float(rightface-.005-offset[:,0].max())]
        if interval[0]>interval[1]:raise RuntimeError('Whole held lateral mouth interval empty')
        old=d.site_xpos[center].copy();outside=old+oR[:,0]*(sum(interval)/2-(oR.T@(old-op))[0]);outside[2]=carry
        segment('radial_lowcarry_mouth_alignment',outside)
        hr=d.site_xmat[hs].reshape(3,3);hp=d.site_xpos[hs]
        local=hr.T@(d.xpos[body]-hp);local[1]=-m.site_size[hs,1]+.004
        goal=hp+hr@local-A@rel[0];goal[2]=carry
        segment('radial_horizontal_first_original_critical',goal)
        if not any(x.get('original_critical')for x in result['route']):raise RuntimeError('Original first-critical not reached')
        result['passed']=True
    except BaseException as exc:
        result['failure']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        result['elapsed_s']=time.monotonic()-start
        result['source_drift']=[name for name,h in pins.items()if hashlib.sha256((Path(name)if Path(name).is_absolute()else ROOT/name).read_bytes()).hexdigest()!=h]
        (H/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ['passed','failure','Scene_constructors','task_controls','elapsed_s']}))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
