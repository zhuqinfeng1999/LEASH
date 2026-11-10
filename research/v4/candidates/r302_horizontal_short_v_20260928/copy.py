"""One continuous own saved-MJB horizontal end-grasp route; zero physics."""
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from research.v4.candidates.r302_horizontal_end_grasp_saved_copy_20260928 import derive
from research.v4.candidates.r3_04_mirrored_initial_saved_copy_20260928.check import no_restart_ik
from research.v4.candidates.r3_04_source_axis_mirrored_low_copy_20260928.check import joint_margins

H=Path(__file__).resolve().parent
SAVED=H/'support_seed0_v2'


def exact_contacts(m,d):
    rows=[]
    for c in d.contact[:d.ncon]:
        names=[m.geom(int(g)).name or '' for g in (c.geom1,c.geom2)]
        if not any(n.startswith(('robot0_','gripper0_','steel_thermos_1_')) for n in names):continue
        pad_can=any(n in derive.PAD_NAMES for n in names) and any(n in derive.CAN_NAMES for n in names)
        rows.append({'geoms':names,'dist_m':float(c.dist),'world_position_m':c.pos.tolist(),
                     'world_frame':c.frame.tolist(),'exact_two_soft_pad_can_material':pad_can,
                     'copied_contact_not_physical_force_evidence':True})
    return rows


def box_vertices(m,d,i):
    signs=np.asarray(list(itertools.product((-1,1),repeat=3)))
    return signs*m.geom_size[i]@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]


def shape_interval(m,d,ids,direction,origin):
    """Exact cylinder support, actual compiled mesh vertices and box corners."""
    lo,hi=math.inf,-math.inf
    for i in ids:
        kind=int(m.geom_type[i]);G=d.geom_xmat[i].reshape(3,3);p=d.geom_xpos[i]-origin
        if kind==5:
            axis=G[:,2];ax=float(np.dot(direction,axis));c=float(np.dot(direction,p))
            span=float(m.geom_size[i,0]*math.sqrt(max(0.,1-ax*ax))+m.geom_size[i,1]*abs(ax))
            a,b=c-span,c+span
        elif kind==6:
            values=(box_vertices(m,d,i)-origin)@direction;a,b=float(values.min()),float(values.max())
        elif kind==7:
            k=int(m.geom_dataid[i]);s=int(m.mesh_vertadr[k]);n=int(m.mesh_vertnum[k])
            values=(m.mesh_vert[s:s+n]@G.T+p)@direction;a,b=float(values.min()),float(values.max())
        else:raise RuntimeError('No guessed extent for own solid geom type '+str(kind))
        lo,hi=min(lo,a),max(hi,b)
    return lo,hi


def main():
    if (H/'copied_report.json').exists():raise RuntimeError('Preserve first continuous copied attempt')
    started=time.monotonic();report={'kind':'own central-short-V horizontal can saved-MJB full route',
      'Scene_constructors':0,'physics_steps':0,'task_controls':0,'GPU_models':0,
      'virtual_relative_private_offline_only':True,'actual_bind_lift_or_method_claim':False,
      'passed':False,'failure':None,'rows':[]}
    m=mujoco.MjModel.from_binary_path(str(SAVED/'compiled_model.mjb'))
    old=np.load(SAVED/'constructor_data.npz',allow_pickle=False);d=mujoco.MjData(m)
    d.qpos[:]=old['qpos'];d.qvel[:]=0.;mujoco.mj_forward(m,d)
    sim=SimpleNamespace(model=SimpleNamespace(_model=m),data=d)
    e=m.site('gripper0_right_grip_site').id;zone=m.site('v42_steel_thermos_1_grasp_zone').id
    body=m.body('steel_thermos_1_main').id;oven=m.body('microwave_1_main').id
    ca=int(m.jnt_qposadr[m.joint('steel_thermos_1_joint0').id])
    arm=[m.joint('robot0_joint'+str(i)).id for i in range(1,8)];adr=m.jnt_qposadr[arm]
    f=[int(m.jnt_qposadr[m.joint(n).id]) for n in ('gripper0_right_finger_joint1','gripper0_right_finger_joint2')]
    qclose=float(m.geom_size[m.geom(derive.CAN_NAMES[0]).id,0])+.0005
    for name in derive.PAD_NAMES:
        i=m.geom(name).id
        if int(m.geom_type[i])!=6 or not np.array_equal(m.geom_size[i],[.008,.004,.008]) or not np.array_equal(m.geom_solref[i],[.01,.5]):
            raise RuntimeError('Actual exact soft-pad metadata changed')
    d.qpos[f]=[.04,-.04];mujoco.mj_forward(m,d)
    home=d.site_xpos[e].copy();homeR=d.site_xmat[e].reshape(3,3).copy()
    p,R=derive.desired_frame(m,d);rel=None
    ik,source=no_restart_ik();(H/'executed_ik_source.py.txt').write_text(source)
    report.update(home_eef_m=home.tolist(),home_R=homeR.tolist(),source_zone_m=p.tolist(),
      source_EEF_R=R.tolist(),source_can_qpos=d.qpos[ca:ca+7].tolist(),
      initial_joint_limits=joint_margins(m,d,arm),source_backoff_m=0.,q_closed_m=qclose,
      source_grasp_precision_m=.0005,source_grasp_precision_rad=math.radians(.5),
      original_task_tolerance_m=[.015,.015,.020],original_global_caps=[1200,90])
    try:
        def pose(stage,u,target,Q,held=False,closed=False,precise=False):
            nonlocal rel
            ans=ik(sim,np.asarray(target),Q,max_iter=280,
                   position_tol=.0005 if precise else .004,
                   rotation_tol=math.radians(.5) if precise else .03)
            success=bool(ans.get('reachable') and ans.get('restart')==0)
            if success:d.qpos[adr]=ans['joint_qpos_rad']
            elif 'best_full_qpos' in ans:d.qpos[:]=ans['best_full_qpos']
            d.qpos[f]=[qclose,-qclose] if closed else [.04,-.04]
            mujoco.mj_forward(m,d)
            A=d.site_xmat[e].reshape(3,3).copy();actual=d.site_xpos[e].copy()
            if held and rel is not None:
                d.qpos[ca:ca+3]=actual+A@rel[0]
                d.qpos[ca+3:ca+7]=Rotation.from_matrix(A@rel[1]).as_quat()[[3,0,1,2]]
                mujoco.mj_forward(m,d)
            contacts=exact_contacts(m,d)
            row={'stage':stage,'u':float(u),'goal_m':np.asarray(target).tolist(),'goal_R':Q.tolist(),
                 'actual_eef_m':d.site_xpos[e].tolist(),'actual_R':d.site_xmat[e].reshape(3,3).tolist(),
                 'ik':ans,'joint_limit_margins':joint_margins(m,d,arm),'finger_qpos_m':d.qpos[f].tolist(),
                 'can_qpos':d.qpos[ca:ca+7].tolist(),'contacts':contacts}
            report['rows'].append(row)
            if not success:
                report['failure']={'kind':'first continuous IK failure/no alternate restart','row':row};return False
            bad=[x for x in contacts if x['dist_m']<-.001 and not x['exact_two_soft_pad_can_material']]
            if bad:
                report['failure']={'kind':'first non-pad full-arm/material/environment contact >1mm','contact':bad[0],'row':row};return False
            if stage=='source_closure_open':
                a=d.site_xpos[m.site('v42_left_pad').id];b=d.site_xpos[m.site('v42_right_pad').id];z=d.site_xpos[zone]
                gap=b-a;distance=np.linalg.norm(gap);along=float(np.dot(z-a,gap/distance));local=A.T@(z-actual)
                eligible=bool(0<=along<=distance and np.all(np.abs(local)<=[.015,.015,.020]))
                row.update(zone_local_error_m=local.tolist(),zone_between_pad_sites=0<=along<=distance,
                           original_assist_eligible=eligible,pad_site_gap_m=float(distance))
                if not eligible:
                    report['failure']={'kind':'real +42mm zone not physically between pads/original tolerance','row':row};return False
                rel=(A.T@(d.xpos[body]-actual),A.T@d.xmat[body].reshape(3,3))
                report['copied_closure_relative']={'position_m':rel[0].tolist(),'R':rel[1].tolist(),'not_actual_weld':True}
            return True

        def segment(stage,start,end,A,B,held=False,closed=False,precise=False):
            a,b=np.asarray(start),np.asarray(end);angle=(Rotation.from_matrix(B@A.T).magnitude())
            n=max(1,math.ceil(np.linalg.norm(b-a)/.020),math.ceil(angle/math.radians(5)))
            sl=Slerp([0,1],Rotation.from_matrix([A,B]))
            for i in range(1,n+1):
                u=i/n
                if not pose(stage,u,a*(1-u)+b*u,sl(u).as_matrix(),held,closed,precise):return False
            return True

        pre=p-R[:,2]*.10
        ok=segment('home_to_horizontal_source_pre',home,pre,homeR,R,precise=True)
        if ok:ok=segment('horizontal_source_approach',pre,p,R,R,precise=True)
        if ok:ok=pose('source_closure_open',1,p,R,precise=True)
        if ok:ok=pose('source_virtual_material_closed',1,p,R,closed=True,precise=True)
        if ok:
            origin=d.site_xpos[e].copy();A=d.site_xmat[e].reshape(3,3).copy()
            op=d.xpos[oven].copy();oR=d.xmat[oven].reshape(3,3).copy()
            entry=np.column_stack((oR[:,0],-oR[:,2],oR[:,1]))
            solids=[i for i in range(m.ngeom) if (m.geom(i).name or '').startswith(('steel_thermos_1_','gripper0_right_'))]
            # Rigid support in the selected entry frame; no object AABB and no
            # R304/R306 extents are inherited. Every cylinder stays analytic.
            vertical=shape_interval(m,d,solids,A@entry.T@np.array([0.,0.,1.]),origin)
            floor=box_vertices(m,d,m.geom('microwave_1_g7').id)[:,2].max()
            roof=box_vertices(m,d,m.geom('microwave_1_g6').id)[:,2].min()
            lo,hi=float(floor+.005-vertical[0]),float(roof-.005-vertical[1])
            carry=max(lo,float(origin[2]+.020))
            report['own_full_can_hand_window']={'floor_world_top_m':float(floor),'roof_world_bottom_m':float(roof),
              'entry_EEF_z_range_m':[lo,hi],'source_20mm_minimum_EEF_z_m':float(origin[2]+.020),
              'selected_lowest_level_m':carry,'shape_count':len(solids),'exact_shape_not_foil_AABB':True,'entry_R':entry.tolist()}
            if carry>hi:raise RuntimeError('Own full can+hand floor/roof window empty')
            lift=origin.copy();lift[2]=carry
            ok=segment('virtual_held_low_lift',origin,lift,A,A,True,True)
            if ok:ok=segment('outside_source_single_mouth_alignment',lift,lift,A,entry,True,True)
            if ok:
                xextent=shape_interval(m,d,solids,oR[:,0],d.site_xpos[e])
                left=((box_vertices(m,d,m.geom('microwave_1_g3').id)-op)@oR)[:,0].max()
                right=((box_vertices(m,d,m.geom('microwave_1_g4').id)-op)@oR)[:,0].min()
                xs=[float(left+.005-xextent[0]),float(right-.005-xextent[1])]
                report['own_lateral_EEF_window_m']=xs
                if xs[0]>xs[1]:raise RuntimeError('Own complete can+closed hand lateral window empty')
                h=m.site('microwave_1_heating_region').id;hp=d.site_xpos[h];hr=d.site_xmat[h].reshape(3,3)
                targetbody=hp-hr[:,1]*(m.site_size[h,1]-.016)
                goal=targetbody-entry@rel[0];local=oR.T@(goal-op);local[0]=sum(xs)/2
                goal=op+oR@local;goal[2]=carry
                outside=lift+oR[:,0]*(sum(xs)/2-(oR.T@(lift-op))[0]);outside[2]=carry
                report['first_original_critical_goal_m']=goal.tolist()
                ok=segment('low_front_lateral_align',lift,outside,entry,entry,True,True)
                if ok:ok=segment('horizontal_first_original_critical',outside,goal,entry,entry,True,True)
                if ok:
                    hlocal=hr.T@(d.xpos[body]-hp)
                    report['copied_original_critical']=bool(np.all(np.abs(hlocal)<m.site_size[h]))
                    report['heating_local_can_center_m']=hlocal.tolist()
                    if not report['copied_original_critical']:raise RuntimeError('Original first critical not actually reached in copy')
        report['passed']=bool(ok and report['failure'] is None)
    except Exception as exc:
        if report['failure'] is None:report['failure']={'kind':type(exc).__name__,'message':str(exc)}
    finally:
        report['elapsed_s']=time.monotonic()-started
        report['source_inputs_sha256']={str((SAVED/n).relative_to(H.parents[3])):hashlib.sha256((SAVED/n).read_bytes()).hexdigest()
                                       for n in ('compiled_model.mjb','constructor_data.npz','report.json')}
        (H/'copied_report.json').write_text(json.dumps(derive.serial(report),indent=2)+'\n')
        np.savez_compressed(H/'last_copied_data.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy())
        print(json.dumps(derive.serial({'passed':report['passed'],'rows':len(report['rows']),
                                     'failure':report['failure'],'elapsed_s':report['elapsed_s']})))


if __name__=='__main__':main()
