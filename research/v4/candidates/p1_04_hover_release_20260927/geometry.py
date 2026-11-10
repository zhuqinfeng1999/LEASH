"""Offline real collision-envelope clearance and copied-data route, no controls."""
import json
import time
from pathlib import Path
from itertools import product
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from research.v4.candidates.p1_04_cutter_holder_20260926.scene import P1CutterFacade
HERE=Path(__file__).resolve().parent

def bounds(m,d,g,R,origin):
    rot=R.T@d.geom_xmat[g].reshape(3,3);center=R.T@(d.geom_xpos[g]-origin);size=m.geom_size[g];kind=m.geom_type[g]
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
        mid=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);points=np.asarray(m.mesh_vert[a:a+n])@rot.T+center
        return points.min(0),points.max(0)
    if kind==mujoco.mjtGeom.mjGEOM_BOX:half=np.abs(rot)@size
    elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:half=size[0]+size[1]*np.abs(rot[:,2])
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:half=size[0]*np.sqrt(np.maximum(0,1-rot[:,2]**2))+size[1]*np.abs(rot[:,2])
    else:raise RuntimeError('Unsupported actual collider '+str(m.geom(g).name))
    return center-half,center+half

def hover(scene):
    m=scene.backend.sim.model._model;d=scene.backend.sim.data._data
    item=scene.backend.objects_dict[scene.item_name];target=scene.backend.objects_dict[scene.target_name]
    bid=m.body(item.root_body).id;tid=m.body(target.root_body).id;sid=m.site('gripper0_right_grip_site').id
    origin=d.xpos[tid].copy();R=d.xmat[tid].reshape(3,3).copy();off=d.xpos[bid]-d.site_xpos[sid]
    if not scene.grasp_assist.held:
        off=d.xpos[bid]-d.site_xpos[m.site('v42_blunt_tool_1_grasp_zone').id]
    goal=origin-off
    copied=mujoco.MjData(m);copied.qpos[:]=d.qpos;mujoco.mj_forward(m,copied)
    current=copied.site_xpos[sid].copy();walls=[m.geom(n).id for n in target.contact_geoms if 'wall'in n]
    surface=[(g,*bounds(m,copied,g,R,origin))for g in walls]
    fingers=[m.jnt_qposadr[m.joint('gripper0_right_finger_joint'+str(i)).id]for i in(1,2)]
    rows=[]
    for gap in(0.,.04):
        copied.qpos[fingers]=gap;mujoco.mj_forward(m,copied)
        for g in range(m.ngeom):
            name=m.geom(g).name or''
            if not(name.startswith('gripper0_')or name=='robot0_link7_collision')or not(m.geom_contype[g]or m.geom_conaffinity[g]):continue
            lo,hi=bounds(m,copied,g,R,current)
            shift=R.T@(goal-origin)
            for other,slo,shi in surface:
                if np.any(hi[:2]+shift[:2]<slo[:2])or np.any(lo[:2]+shift[:2]>shi[:2]):continue
                z=float(origin[2]+shi[2]+.005-lo[2])
                rows.append(dict(geom=name,wall=m.geom(other).name,gap_m=gap,required_eef_z_m=z))
    if not rows:raise RuntimeError('Actual wall clearance envelope missing')
    controlling=max(rows,key=lambda row:row['required_eef_z_m']);goal[2]=controlling['required_eef_z_m']
    item_bounds=[bounds(m,d,m.geom(n).id,R,d.xpos[bid])for n in item.contact_geoms]
    local_body=R.T@(goal+off-origin);lo=np.min([b[0]for b in item_bounds],0)+local_body;hi=np.max([b[1]for b in item_bounds],0)+local_body
    inside=bool(np.all(lo[:2]>=np.array([-.087,-.062]))and np.all(hi[:2]<=np.array([.087,.062]))and -.007001<=lo[2]<=.102501)
    proof=dict(kind='single conservative real gripper/link7 wall envelope minimum5mm, no height grid',goal_eef_m=goal.tolist(),body_goal_m=(goal+off).tolist(),item_actual_offset_world_m=off.tolist(),controlling=controlling,all_constraints=rows,
        full_shape_local_bounds_m=[lo.tolist(),hi.tolist()],preopen_entire_xy_inside=bool(np.all(lo[:2]>=[-.087,-.062])and np.all(hi[:2]<=[.087,.062])),original_target_region_now=inside,
        bottom_above_floor_m=float(lo[2]+.005),bottom_below_rim_m=float(.1025-lo[2]))
    if not inside:raise RuntimeError('Derived lowest clear hover does not satisfy original actual pre-open target region: '+json.dumps(proof))
    return goal,proof

def main():
    dest=HERE/'copied_route_actual_stage_bound.json'
    if dest.exists():raise RuntimeError('Preserve first geometry result')
    started=time.monotonic();scene=P1CutterFacade(0);rows=[];failure=None
    try:
        m=scene.backend.sim.model._model;actual=scene.backend.sim.data._data;d=mujoco.MjData(m);d.qpos[:]=actual.qpos;mujoco.mj_forward(m,d)
        goal,proof=hover(scene);sid=m.site('gripper0_right_grip_site').id;desiredR=d.site_xmat[sid].reshape(3,3).copy()
        qi=m.jnt_qposadr[[m.joint('robot0_joint'+str(i)).id for i in range(1,8)]]
        limits=m.jnt_range[[m.joint('robot0_joint'+str(i)).id for i in range(1,8)]]
        item=scene.backend.objects_dict[scene.item_name];bid=m.body(item.root_body).id;adr=m.jnt_qposadr[m.joint(item.joints[0]).id]
        grasp=d.site_xpos[m.site('v42_blunt_tool_1_grasp_zone').id].copy();off=np.array(proof['item_actual_offset_world_m']);original_q=d.qpos[adr:adr+7].copy()
        target=np.array(proof['body_goal_m']);rear=-.35-off[0]
        stages=[('source_staging',np.array([-.25,grasp[1],1.18]),False),('above_handle',np.array([grasp[0],grasp[1],1.18]),False),('approach_handle',np.array([grasp[0],grasp[1],1.035]),False),('lower_to_handle',grasp,False),('lift',np.array([grasp[0],grasp[1],1.09]),True),('rear_detour',np.array([rear,grasp[1],1.16]),True),('rear_across',np.array([rear,target[1]-off[1],1.16]),True),('above_holder',np.array([target[0]-off[0],target[1]-off[1],1.16]),True),('clear_hover_release',goal,True)]
        robot=[g for g in range(m.ngeom)if(m.geom(g).name or'').startswith(('robot0_','gripper0_'))and(m.geom_contype[g]or m.geom_conaffinity[g])]
        own=[m.geom(n).id for n in item.contact_geoms]
        outside=[g for g in range(m.ngeom)if g not in robot+own and(m.geom_contype[g]or m.geom_conaffinity[g])]
        for stage,end,held in stages:
            start=d.site_xpos[sid].copy()
            for frac in np.linspace(0,1,13)[1:]:
                point=start*(1-frac)+end*frac
                def residual(q):
                    d.qpos[qi]=q;mujoco.mj_forward(m,d)
                    return np.r_[d.site_xpos[sid]-point,.2*Rotation.from_matrix(desiredR@d.site_xmat[sid].reshape(3,3).T).as_rotvec()]
                fit=least_squares(residual,np.clip(d.qpos[qi],limits[:,0]+.005,limits[:,1]-.005),bounds=(limits[:,0]+.005,limits[:,1]-.005),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                residual(fit.x)
                d.qpos[adr:adr+7]=original_q
                if held:d.qpos[adr:adr+3]=d.site_xpos[sid]+off
                mujoco.mj_forward(m,d);contacts=[]
                for a in robot+(own if held else[]):
                    for b in outside:
                        if m.body_weldid[m.geom_bodyid[a]]==m.body_weldid[m.geom_bodyid[b]]or not((m.geom_contype[a]&m.geom_conaffinity[b])or(m.geom_contype[b]&m.geom_conaffinity[a])):continue
                        dist=float(mujoco.mj_geomDistance(m,d,a,b,.02,None))
                        if dist< -1e-7:contacts.append(dict(first=m.geom(a).name,second=m.geom(b).name,distance_m=dist))
                miss=float(np.linalg.norm(d.site_xpos[sid]-point));angle=float(np.linalg.norm(Rotation.from_matrix(desiredR@d.site_xmat[sid].reshape(3,3).T).as_rotvec()))
                original_tolerance=.005 if stage=='lower_to_handle' else .008
                actual_stage_bound=max(.018,2*original_tolerance)
                row=dict(stage=stage,fraction=float(frac),position_error_m=miss,orientation_error_rad=angle,contacts=contacts,qpos=d.qpos.tolist(),original_OSC_loop_tolerance_m=original_tolerance,original_actual_stage_failure_bound_m=actual_stage_bound,passed=miss<=actual_stage_bound and not contacts);rows.append(row)
                if not row['passed']:failure={k:row[k]for k in ['stage','fraction','position_error_m','orientation_error_rad','contacts']};break
            if failure:break
        final_bounds=None
        if failure is None:
            target_obj=scene.backend.objects_dict[scene.target_name];tid=m.body(target_obj.root_body).id;origin=d.xpos[tid];R=d.xmat[tid].reshape(3,3)
            shapes=[bounds(m,d,g,R,origin)for g in own];lo=np.min([v[0]for v in shapes],0);hi=np.max([v[1]for v in shapes],0)
            inside=bool(np.all(lo[:2]>=[-.087,-.062])and np.all(hi[:2]<=[.087,.062])and -.007001<=lo[2]<=.102501)
            final_bounds=dict(full_collision_shape_local_bounds_m=[lo.tolist(),hi.tolist()],original_target_region=inside)
            if not inside:failure=dict(stage='actual_terminal_preopen_shape',reason='actual copied full shape not in original target region')
        report=dict(kind='copied-data IK/actual collider witness only; zero controller commands',passed=failure is None,failure=failure,poses=len(rows),rows=rows,derived_hover=proof,final_actual_bounds=final_bounds,elapsed_s=time.monotonic()-started)
    except Exception as exc:report=dict(passed=False,error=f'{type(exc).__name__}: {exc}',rows=rows,elapsed_s=time.monotonic()-started)
    finally:scene.close()
    dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='rows'},indent=2));return 0 if report['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
