"""One geometry-derived external-palm side grip; zero Scene/live controls."""
from pathlib import Path
import hashlib
import json
import time

import mujoco
import numpy as np

from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.r305_single_ceramic_lug_20260928 import check as original
from research.v4.candidates.r305_seed1_panel_readonly_20260928.analyze import vertices, bbox

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SAVED=HERE.parent/'r305_device_cart_y80_geometry_20260928'
PADS=tuple(original.parent.PADS)


def side_material(m,d,center,body,left,right,finger):
    # Reuse the exact existing material rays/pad-width calculation, not its
    # world-down pose. Only one side-frame sign is derived from the hard palm.
    _,_,closed,prior=original.lug_material(m,d,center,body,left,right,finger)
    E=d.site_xmat[center].reshape(3,3);p=d.site_xpos[center].copy()
    palm=m.geom('gripper0_right_hand_collision').id
    local=(vertices(m,d,palm)-p)@E
    if not np.all(local[:,2]<0):raise RuntimeError('Palm envelope does not determine unique external-palm sign')
    B=d.xmat[body].reshape(3,3)
    outward=B[:,0].copy();outward/=np.linalg.norm(outward)
    jaw=B[:,1].copy();jaw/=np.linalg.norm(jaw)
    inward=-outward;R=np.column_stack((jaw,np.cross(inward,jaw),inward))
    material=d.site_xpos[m.site('v42_fork_bowl_1_grasp_zone').id].copy()
    lug=m.geom('fork_bowl_1_ceramic_lug_collision').id
    normal_ray=float(mujoco.mj_rayMesh(m,d,lug,material,outward))
    negative_ray=float(mujoco.mj_rayMesh(m,d,lug,material,-jaw))
    positive_ray=float(mujoco.mj_rayMesh(m,d,lug,material,jaw))
    if min(normal_ray,negative_ray,positive_ray)<=0:raise RuntimeError('Actual solid ear material ray failed')
    material+=(positive_ray-negative_ray)/2*jaw
    lp=E.T@(d.site_xpos[left]-p);rp=E.T@(d.site_xpos[right]-p)
    midpoint=(lp+rp)/2
    pad=m.geom(PADS[0]).id
    half=float(np.sum(np.abs((E.T@d.geom_xmat[pad].reshape(3,3))[0])*m.geom_size[pad]))
    zero=[abs(float(lp[0]))-float(d.qpos[m.jnt_qposadr[finger[0]]]),
          abs(float(rp[0]))-abs(float(d.qpos[m.jnt_qposadr[finger[1]]]))]
    closed=(negative_ray+positive_ray)/2+half-sum(zero)/2
    if not 0<closed<=.04:raise RuntimeError('Exact material-width closure outside original finger limits')
    target=material-R@midpoint
    return R,target,closed,dict(version='R305-true-ear-horizontal-inwardZ-external-palm-v1',
        only_expert_grip_changed=True,asset_grasp_region_changed=False,
        body_outward_normal=outward.tolist(),EEF_Z_material_inward=inward.tolist(),jaw_axis_world=jaw.tolist(),
        hard_palm_EEF_local_bbox=bbox(local),sign_derived_from_entire_negative_palm_Z=True,
        no_two_sign_pose_or_angle_trial=True,actual_normal_ray_m=normal_ray,
        actual_tangent_material_ray_m=[negative_ray,positive_ray],material_width_m=negative_ray+positive_ray,
        actual_pad_halfwidth_m=half,pad_midpoint_EEF_local=midpoint.tolist(),closed_qpos=[closed,-closed],
        target_EEF=target.tolist(),target_R=R.tolist(),unchanged_zone_local=m.site_pos[m.site('v42_fork_bowl_1_grasp_zone').id].tolist(),
        grasp_tolerance_xyz=[.015,.015,.020],prior_material_source=prior['actual_material_mesh'])


def static_frame(m,d,kind,R,target,qs,center,left,right,zone,finger):
    for j,q in zip(finger,qs):d.qpos[m.jnt_qposadr[j]]=q
    mujoco.mj_forward(m,d)
    oldp=d.site_xpos[center].copy();oldR=d.site_xmat[center].reshape(3,3).copy()
    hand=[g for g in range(m.ngeom)if (m.geom(g).name or '').startswith('gripper0_')
          and(m.geom_contype[g]or m.geom_conaffinity[g])]
    source_sites={i:(d.site_xpos[i].copy(),d.site_xmat[i].reshape(3,3).copy())for i in (center,left,right)}
    local_bounds=[]
    for g in hand:
        local_bounds.append(dict(geom=m.geom(g).name,EEF_local_bbox=bbox((vertices(m,d,g)-oldp)@oldR)))
        localp=oldR.T@(d.geom_xpos[g]-oldp);localR=oldR.T@d.geom_xmat[g].reshape(3,3)
        d.geom_xpos[g]=target+R@localp;d.geom_xmat[g]=(R@localR).ravel()
    # Direct transformed compiled shapes only; no whole-arm reach/FK claim.
    for i,(p,Q)in source_sites.items():
        d.site_xpos[i]=target+R@(oldR.T@(p-oldp));d.site_xmat[i]=(R@oldR.T@Q).ravel()
    helper=GraspAssist.__new__(GraspAssist);helper.model=m;helper.data=d
    helper._center=center;helper._left=left;helper._right=right
    eligible=bool(helper._eligible(zone,(.015,.015,.020)))
    obstacles=[g for g in range(m.ngeom)if (m.geom_contype[g]or m.geom_conaffinity[g]) and
        (m.geom(g).name=='table_collision' or(m.geom(g).name or '').startswith(('fork_bowl_1_','bowl_board_1_','microwave_1_')))]
    rows=[]
    for a in hand:
        for b in obstacles:
            points=np.zeros(6);distance=float(mujoco.mj_geomDistance(m,d,a,b,.10,points))
            if distance>=.10:continue
            names=[m.geom(a).name,m.geom(b).name]
            pad=names[0]in PADS and names[1].startswith('fork_bowl_1_ceramic_')and names[1].endswith('_collision')
            rows.append(dict(geoms=names,dist_m=distance,nearest_points=points.tolist(),
                             exact_pad_material=pad,hard_stop=bool(distance<-.001 and not pad)))
    return dict(kind=kind,target_EEF=target.tolist(),target_R=R.tolist(),finger_qpos=qs,
        actual_compiled_material_eligible=eligible,contacts=rows,hard_geometry_local_bounds=local_bounds,
        scope='Rigid compiled hand assembly at proposed material pose; no IK/joint/physics qualification')


def main():
    if (HERE/'report.json').exists():raise RuntimeError('Single normal side-grip observation')
    pins=json.loads((HERE/'source_freeze.json').read_text())
    for n,s in pins.items():
        p=Path(n)if n.startswith('/')else ROOT/n
        if hashlib.sha256(p.read_bytes()).hexdigest()!=s:raise RuntimeError('Source drift '+n)
    started=time.monotonic();out=dict(passed=False,failure=None,frames=[],source_sha256=pins,
        Scene_constructors=0,new_model_compiles=0,actual_task_controls=0,GPU_services_models=0,IK_calls=0)
    try:
        m=mujoco.MjModel.from_binary_path(str(SAVED/'lug_model.mjb'));saved=np.load(SAVED/'after_neutral_data.npz')
        d=mujoco.MjData(m);d.qpos[:]=saved['qpos'];d.qvel[:]=saved['qvel'];mujoco.mj_forward(m,d)
        center=m.site('gripper0_right_grip_site').id;left=m.site('v42_left_pad').id;right=m.site('v42_right_pad').id
        zone=m.site('v42_fork_bowl_1_grasp_zone').id;body=m.body('fork_bowl_1_main').id
        finger=[m.joint('gripper0_right_finger_joint1').id,m.joint('gripper0_right_finger_joint2').id]
        initial=d.qpos.copy();R,target,closed,out['design']=side_material(m,d,center,body,left,right,finger)
        for kind,qs in [('actual_open',[float(initial[m.jnt_qposadr[j]])for j in finger]),('material_width_closed',[closed,-closed])]:
            d.qpos[:]=initial
            frame=static_frame(m,d,kind,R,target,qs,center,left,right,zone,finger)
            out['frames'].append(frame)
            if any(c['hard_stop']for c in frame['contacts']):raise RuntimeError('First fixed side-lug '+kind+' hard palm/finger/environment intersection>1mm')
            if kind=='material_width_closed'and not frame['actual_compiled_material_eligible']:
                raise RuntimeError('Actual material-width closure not eligible under original tolerance')
        out['passed']=True
    except Exception as exc:out['failure']=dict(type=type(exc).__name__,message=str(exc))
    out['elapsed_s']=time.monotonic()-started
    (HERE/'report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(passed=out['passed'],failure=out['failure'],frames=len(out['frames']),
        hard=[c for f in out['frames']for c in f['contacts']if c['hard_stop']],IK_calls=out['IK_calls']),indent=2))
    return 0 if out['passed']else 2


if __name__=='__main__':raise SystemExit(main())
