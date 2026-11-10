"""One actual-held saved-state source-clearance geometry query; zero physics."""
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
from research.v4.candidates.r302_condim6_top_mirror_tall350_saved_copy_20260928.check import all_contacts
from research.v4.candidates.r302_horizontal_end_grasp_saved_copy_20260928 import derive
from research.v4.candidates.r302_horizontal_short_v_20260928.copy import shape_interval, box_vertices
from research.v4.candidates.r3_04_source_axis_mirrored_low_copy_20260928.check import joint_margins

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
RAW = ROOT / 'research/v4/candidates/r302_seeded_model_derived_guard_v2_20260928/root_qualification_seed1_9/naive_seed1'


def admit():
    pins = json.loads((H/'source_freeze_before.json').read_text())['source_sha256']
    for name, digest in pins.items():
        p = Path(name) if Path(name).is_absolute() else ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Exact source admission failed '+name)
    return len(pins)


def radius_upper(m, d, ids, origin):
    radii = []
    for i in ids:
        Q=d.geom_xmat[i].reshape(3,3); p=d.geom_xpos[i]-origin
        if int(m.geom_type[i])==5:
            axial=float(p @ Q[:,2]); transverse=float(np.linalg.norm(p-axial*Q[:,2]))
            value=math.hypot(abs(axial)+m.geom_size[i,1],transverse+m.geom_size[i,0])
        elif int(m.geom_type[i])==6:
            value=float(np.linalg.norm(box_vertices(m,d,i)-origin,axis=1).max())
        elif int(m.geom_type[i])==7:
            k=int(m.geom_dataid[i]);a=int(m.mesh_vertadr[k]);n=int(m.mesh_vertnum[k])
            v=m.mesh_vert[a:a+n].astype(float)@Q.T+p
            value=float(np.linalg.norm(v,axis=1).max())
        else:raise RuntimeError('No guessed radius for full held solid '+m.geom(i).name)
        radii.append((value,m.geom(i).name))
    return max(radii)


def main():
    if (H/'copied_report.json').exists():raise RuntimeError('Preserve first copied attempt')
    count=admit();started=time.monotonic()
    raw=json.loads((RAW/'result.json').read_text())
    row=next(r for r in raw['actual_control_rows']if r['sequence']==272)
    original_lateral=next(r for r in raw['actual_control_rows']if r['sequence']==273)
    m=mujoco.MjModel.from_binary_path(str(RAW/'startup_seed0/compiled_model.mjb'))
    old_eq=m.eq_data.copy()
    weld=m.equality('v42_steel_thermos_1_assist_weld').id
    # Copy the actual recorded bind annotation into the detached model only.
    m.eq_data[weld]=raw['actual_closure_relative']['weld_data']
    if any(not np.array_equal(old_eq[i],m.eq_data[i])for i in range(m.neq)if i!=weld):
        raise RuntimeError('Unexplained independent model mutation')
    d=mujoco.MjData(m);d.qpos[:]=row['qpos'];d.qvel[:]=row['qvel'];d.ctrl[:]=row['ctrl'];d.eq_active[:]=row['eq_active'];d.time=row['time_s']
    mujoco.mj_forward(m,d)
    e=m.site('gripper0_right_grip_site').id;body=m.body('steel_thermos_1_main').id
    oven=m.body('microwave_1_main').id;ca=int(m.jnt_qposadr[m.joint('steel_thermos_1_joint0').id])
    arms=[m.joint('robot0_joint'+str(i)).id for i in range(1,8)];adr=m.jnt_qposadr[arms]
    fingers=[int(m.jnt_qposadr[m.joint(n).id])for n in ['gripper0_right_finger_joint1','gripper0_right_finger_joint2']]
    qf=d.qpos[fingers].copy()
    origin=d.site_xpos[e].copy();A=d.site_xmat[e].reshape(3,3).copy()
    rel=(A.T@(d.xpos[body]-origin),A.T@d.xmat[body].reshape(3,3))
    target_R=np.asarray(row['goal_R'],float)
    cans=[m.geom(n).id for n in derive.CAN_NAMES]
    solids=cans+derive.hand_geoms(m)
    rails=[m.geom('r302_v_rail_'+s).id for s in ['-1','1']]
    source_geoms=rails+[m.geom('r302_v_base').id]
    crest=max(float(box_vertices(m,d,i)[:,2].max())for i in source_geoms)
    full_lower=shape_interval(m,d,solids,np.asarray([0.,0.,1.]),origin)[0]
    radius,radius_geom=radius_upper(m,d,solids,origin)
    orientation_margin=2*radius*math.sin(.03/2)
    high=origin[2]+crest-(origin[2]+full_lower)+.005+.008+orientation_margin
    oR=d.xmat[oven].reshape(3,3).copy();op=d.xpos[oven].copy()
    floor=float(box_vertices(m,d,m.geom('microwave_1_g7').id)[:,2].max())
    roof=float(box_vertices(m,d,m.geom('microwave_1_g6').id)[:,2].min())
    zrange=shape_interval(m,d,solids,np.asarray([0.,0.,1.]),origin)
    window=[floor+.005-zrange[0],roof-.005-zrange[1]]
    old_low=float(original_lateral['goal'][2])
    lift=origin.copy();lift[2]=high
    outside=np.asarray(original_lateral['goal'],float).copy();outside[2]=high
    low=outside.copy();low[2]=old_low
    sim=SimpleNamespace(model=SimpleNamespace(_model=m),data=d)
    report=dict(kind='ONE actual seed1 seq272 full-held source-crest clearance copied route',
        source_count=count, Scene_constructors=0, task_controls=0, physics_steps=0, GPU_services_models=0,
        start_actual_sequence=272, actual_prefix_replayed=False, passed=False,failure=None,rows=[],
        independent_actual_bind_row=weld, independent_eq_rows_changed=np.flatnonzero(np.any(old_eq!=m.eq_data,axis=1)).tolist(),
        actual_lift_m=row['evaluator']['metal_item_center_m'][2]-raw['initial_truth']['metal_item_center_m'][2],
        actual_held=row['held'], cached_EEF_m=row['EEF'], fresh_FK_EEF_m=origin.tolist(),
        cached_fresh_EEF_delta_m=(origin-np.asarray(row['EEF'])).tolist(),
        cached_fresh_R_delta_rad=float(Rotation.from_matrix(A@np.asarray(row['EEF_R']).T).magnitude()),
        original_joint_margins=joint_margins(m,d,arms), original_caps=[1200,90],
        original_ik_signature=str(inspect.signature(_ik)),continuous_restart0_required=True,
        real_control_or_OSC_or_task_qualification=False,
        geometry=dict(V_crest_world_z_m=crest,full_held_lower_relative_z_m=full_lower,
            full_held_radius_bound_m=radius,limiting_radius_geom=radius_geom,
            existing_rotation_tolerance_rad=.03,rotation_excursion_margin_m=orientation_margin,
            existing_position_stop_error_m=.008,source_clearance_m=.005,
            unique_source_clearance_EEF_height_m=high,old_low_carry_m=old_low,
            oven_floor_world_z_m=floor,oven_roof_world_z_m=roof,current_full_held_EEF_height_window_m=window),
        actual_start_relative=dict(position_m=rel[0].tolist(),R=rel[1].tolist(),not_installed_live=True),
        start_contacts=all_contacts(m,d))
    source_lines,source_start=inspect.getsourcelines(_ik)
    record_line=source_start+next(i for i,line in enumerate(source_lines)if 'if best is None or' in line)

    def solve(p,R):
        evaluations=[]
        def trace(frame,event,arg):
            if frame.f_code is _ik.__code__ and event=='line' and frame.f_lineno==record_line:
                v=frame.f_locals;evaluations.append(dict(restart=int(v['restart']),iteration=int(v['iteration']),
                    qpos=v['data'].qpos.copy(),position_error_m=float(v['pos_error']),rotation_error_rad=float(v['rot_error'])))
            return trace
        before=sys.gettrace()
        try:sys.settrace(trace);ans=_ik(sim,np.asarray(p),R)
        finally:sys.settrace(before)
        return ans,evaluations

    def pose(stage,u,p,R):
        ans,ev=solve(p,R)
        if ans.get('reachable'):d.qpos[adr]=ans['joint_qpos_rad']
        elif ev:d.qpos[:]=min(ev,key=lambda v:v['position_error_m']+.06*v['rotation_error_rad'])['qpos']
        d.qpos[fingers]=qf;mujoco.mj_forward(m,d)
        E=d.site_xpos[e].copy();Q=d.site_xmat[e].reshape(3,3).copy()
        d.qpos[ca:ca+3]=E+Q@rel[0]
        d.qpos[ca+3:ca+7]=Rotation.from_matrix(Q@rel[1]).as_quat()[[3,0,1,2]]
        mujoco.mj_forward(m,d)
        cs=all_contacts(m,d)
        line=dict(stage=stage,u=float(u),goal_m=np.asarray(p).tolist(),goal_R=R.tolist(),ik=ans,
            actual_EEF_m=E.tolist(),actual_R=Q.tolist(),qpos=d.qpos.tolist(),finger_qpos_m=qf.tolist(),
            joint_margins=joint_margins(m,d,arms),can_qpos=d.qpos[ca:ca+7].tolist(),contacts=cs,
            exact_full_held_vertical_bounds_m=list(shape_interval(m,d,solids,np.array([0.,0.,1.]),np.zeros(3))),
            copied_contact_not_actual_force=True,all_original_ik_evaluations=ev)
        report['rows'].append(line)
        if not ans.get('reachable')or ans.get('restart')!=0:
            report['failure']=dict(kind='first original IK failure/noncontinuous restart',row=line);return False
        bad=[c for c in cs if c['hard_stop']]
        if bad:report['failure']=dict(kind='first nonpad/nonsupport hard contact>1mm',contact=bad[0],row=line);return False
        return True

    def segment(stage,a,b,R,C):
        a,b=np.asarray(a),np.asarray(b)
        count=max(1,math.ceil(np.linalg.norm(b-a)/.020),math.ceil(Rotation.from_matrix(C@R.T).magnitude()/math.radians(5)))
        sl=Slerp([0,1],Rotation.from_matrix([R,C]))
        for k in range(1,count+1):
            u=k/count
            if not pose(stage,u,(1-u)*a+u*b,sl(u).as_matrix()):return False
        return True

    try:
        if not row['held']or report['actual_lift_m']<.020:raise RuntimeError('Original seq272 was not true held20mm')
        if any(c['hard_stop']for c in report['start_contacts']):raise RuntimeError('Recorded start has nonpad/nonsupport hard gap')
        if high>window[1]:raise RuntimeError('Unique clearance exceeds unchanged full-shape roof window')
        ok=segment('source_fullshape_clearance_lift',origin,lift,A,target_R)
        if ok:ok=segment('above_V_lateral_mouth_alignment',lift,outside,target_R,target_R)
        if ok:
            source_y=shape_interval(m,d,source_geoms,oR[:,0],np.zeros(3))
            payload_y=shape_interval(m,d,solids,oR[:,0],np.zeros(3))
            separation=max(payload_y[0]-source_y[1],source_y[0]-payload_y[1])
            report['source_clear_before_descent']=dict(axis=oR[:,0].tolist(),source_interval_m=list(source_y),
                held_interval_m=list(payload_y),signed_separation_m=separation)
            if separation<.005:raise RuntimeError('Full held shape not outside V before descent')
            ok=segment('mouth_outside_restore_original_low',outside,low,target_R,target_R)
        if ok:
            # Existing copied endpoint16mm/default12mm is kept separate from
            # the unchanged actual-driver4mm/final1mm. This is not actual motion.
            current=d.site_xpos[e].copy();xs=shape_interval(m,d,solids,oR[:,0],current)
            left=float(((box_vertices(m,d,m.geom('microwave_1_g3').id)-op)@oR[:,0]).max())
            right=float(((box_vertices(m,d,m.geom('microwave_1_g4').id)-op)@oR[:,0]).min())
            X=[left+.005-xs[0],right-.005-xs[1]]
            if X[0]>X[1]:raise RuntimeError('Own complete closedhand/can mouth lateral window empty')
            hs=m.site('microwave_1_heating_region').id;hp=d.site_xpos[hs].copy();hR=d.site_xmat[hs].reshape(3,3).copy()
            body_goal=hp-hR[:,1]*(m.site_size[hs,1]-.016)
            goal=body_goal-target_R@rel[0];local=oR.T@(goal-op);local[0]=sum(X)/2
            goal=op+oR@local;goal[2]=old_low
            report['original_first_critical_goal_m']=goal.tolist();report['original_lateral_window_m']=X
            report['endpoint_protocol']=dict(copied_inside_margin_m=.016,copied_original_IK_position_tol_m=.012,
                actual_driver_inside_margin_m=.004,actual_driver_final_position_tol_m=.001,
                copied_does_not_qualify_actual_endpoint=True)
            ok=segment('original_first_critical_entry',low,goal,target_R,target_R)
            if ok:
                hlocal=hR.T@(d.xpos[body]-hp)
                critical=bool(np.all(np.abs(hlocal)<m.site_size[hs]))
                report['original_center_critical_geometry']=dict(local=hlocal.tolist(),site_size=m.site_size[hs].tolist(),critical=critical)
                if not critical:raise RuntimeError('Original critical not reached in existing16mm copied diagnostic')
        report['passed']=bool(ok and report['failure']is None)
    except Exception as exc:
        if report['failure']is None:report['failure']=dict(kind=type(exc).__name__,message=str(exc))
    finally:
        report['elapsed_s']=time.monotonic()-started
        (H/'copied_report.json').write_text(json.dumps(derive.serial(report),indent=2)+'\n')
        np.savez_compressed(H/'last_copied_data.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),eq_active=d.eq_active.copy())
        print(json.dumps(derive.serial(dict(passed=report['passed'],failure=report['failure'],rows=len(report['rows']),geometry=report['geometry'],elapsed_s=report['elapsed_s'])),indent=2))


if __name__=='__main__':main()
