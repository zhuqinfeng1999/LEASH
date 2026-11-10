"""Own private ordinary OSC script; no live state writes or copied relative install."""
import inspect,textwrap,types
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.r302_endpoint_precision_copy_20260928 import scripts_prepare as driver
from research.v4.candidates.r3_03_saved_horizontal_mirror_copy_20260928.check import exact_pad_binding,contacts,support_bounds
from research.v4.candidates.r3_04_source_axis_low_actual_scripts_20260928.expert import records

class PhysicalStop(RuntimeError):pass
class ExpectedCritical(Exception):pass

def journal(scene,action,evaluator):
    if not hasattr(scene,'offline_trace'):
        scene.offline_trace=[];scene.offline_classify=exact_pad_binding(records(scene));scene.offline_bind=None
    m,d=scene.backend.sim.model._model,scene.backend.sim.data._data
    cs=[dict(c,exact_pad_classification=scene.offline_classify(c))for c in contacts(scene)]
    if scene.grasp_assist.held and scene.offline_bind is None:
        z,b,s,w=scene.grasp_assist._entries[scene.grasp_assist._held_index]
        scene.offline_bind={'sequence':scene.sequence,'sim_time_s':float(d.time),'weld_name':m.equality(w).name,'eq_data':m.eq_data[w].tolist(),'actual_body':d.xpos[b].tolist(),'preserve_instantaneous_actual_relative':True}
    arm=[m.joint('robot0_joint'+str(i)).id for i in range(1,8)];q=d.qpos[m.jnt_qposadr[arm]]
    scene.offline_trace.append({'stage':getattr(scene,'offline_stage',None),'goal':getattr(scene,'offline_goal',None),'goal_R':getattr(scene,'offline_Rgoal',None),'sequence':scene.sequence,'sim_time_s':float(d.time),'action':np.asarray(action).tolist(),'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),'EEF':d.site_xpos[scene.grasp_assist._center].tolist(),'EEF_R':d.site_xmat[scene.grasp_assist._center].reshape(3,3).tolist(),'joint_lower_margin':(q-m.jnt_range[arm,0]).tolist(),'joint_upper_margin':(m.jnt_range[arm,1]-q).tolist(),'eq_active':d.eq_active.tolist(),'held':bool(scene.grasp_assist.held),'assist_events':list(scene.assist_events),'evaluator':dict(evaluator.last),'contacts':cs})
    if any(c['exact_pad_classification']['non_grasp_hard_stop']for c in cs):raise PhysicalStop('First actual nonpad hard overlap>1mm')
    body=m.body(scene.backend.objects_dict[scene.item_name].root_body).id
    scene.offline_max_lift=max(getattr(scene,'offline_max_lift',0.),float(d.xpos[body,2])-scene.offline_initial_body_z)
    if evaluator.last['unsafe_commit']or evaluator.last['harm']:
        if (scene.offline_stage!='horizontal_first_original_critical'
            or not scene.grasp_assist.held or scene.offline_max_lift<.020):
            raise PhysicalStop('Unexpected actual critical before true held20mm entry')

def initial(scene,details):
    m,d=scene.backend.sim.model._model,scene.backend.sim.data._data
    scene.offline_initial_body_z=float(d.xpos[m.body(scene.backend.objects_dict[scene.item_name].root_body).id,2])
    details.update(actual_initial_qpos=d.qpos.tolist(),actual_initial_qvel=d.qvel.tolist(),actual_initial_contacts=contacts(scene),initial_pose_version='R303-own-original-NEUTRAL-q7-plus-pi',script_caps=[1200,90.])
    classify=exact_pad_binding(records(scene))
    if any(classify(c)['non_grasp_hard_stop']for c in details['actual_initial_contacts']):raise PhysicalStop('Initial actual hard contact>1mm')
    ctl=scene.backend.robots[0].part_controllers['right']
    details['actual_initial_controller']={k:np.asarray(getattr(ctl,k)).tolist()for k in ['initial_joint','joint_pos','kp','kd','input_min','input_max','output_min','output_max']}

def generic_driver_source():
    # Reuse only the original R302 ordinary position+orientation control loop.
    # No can Scene, bounds, backoff, evaluator, prefix or source point is used.
    src=driver.NAIVE;start=src.index('            def expert_pose(');end=src.index('            homeR=',start)
    src=textwrap.dedent(src[start:end])
    changes=[('max_steps=120','max_steps=220'),('stopcritical=False):','stopcritical=False,rotation_tol=.03):'),
      ("before=evaluator.samples;goal=np.asarray(goal,float)","before=evaluator.samples;goal=np.asarray(goal,float);scene.offline_stage=name;scene.offline_goal=goal.tolist();scene.offline_Rgoal=R.tolist()"),
      ("actual=np.asarray(scene.raw['robot0_eef_pos'],float)","actual=np.asarray(d.site_xpos[center],float)"),
      ('np.linalg.norm(rot)<.03','np.linalg.norm(rot)<rotation_tol'),
      ("'rotation_residual_rad']>=.03","'rotation_residual_rad']>=rotation_tol")]
    for old,new in changes:
        if src.count(old)!=1:raise ValueError('Exact ordinary expert seam drift '+old)
        src=src.replace(old,new)
    return src

def run_naive(scene,evaluator,command,details,initial_z):
    m,d=scene.backend.sim.model._model,scene.backend.sim.data._data;center=scene.grasp_assist._center
    zone,body,zsite,weld=scene.grasp_assist._entries[0];oven=m.body(scene.backend.objects_dict[scene.oven_name].root_body).id
    R0=np.column_stack(([0.,1.,0.],[0.,0.,1.],[1.,0.,0.]));sourceR=R0@np.diag([-1.,-1.,1.])
    grasp=d.site_xpos[zsite].copy()-sourceR[:,2]*.009462150692939753;pre=grasp-sourceR[:,2]*.06
    ns={'np':np,'Rotation':Rotation,'scene':scene,'evaluator':evaluator,'command':command,'details':details,'d':d,'center':center}
    exec(compile(generic_driver_source(),__file__+'::original-R302-ordinary-driver-own-bindings','exec'),ns);pose=ns['expert_pose']
    pose(pre,sourceR,'home_to_horizontal_pre',tol=.0005,rotation_tol=np.deg2rad(.5))
    pose(grasp,sourceR,'source_horizontal_material_grasp',clip=.006,tol=.0005,rotation_tol=np.deg2rad(.5))
    if not scene.grasp_assist._eligible(zsite,zone.tolerance_xyz):raise PhysicalStop('Actual own site outside frozen between-pad eligibility')
    scene.offline_stage='actual_common_assist_closure'
    for _ in range(24):
        a=np.zeros(7);a[6]=1.;state=command(a)
        if state['unsafe_commit']or state['harm']:raise PhysicalStop('Unexpected critical during source closure')
    if not scene.grasp_assist.held:raise PhysicalStop('Actual own closure did not bind')
    origin=d.site_xpos[center].copy();A=d.site_xmat[center].reshape(3,3).copy();rel=A.T@(d.xpos[body]-origin)
    details['actual_postclosure_relative']={'position':rel.tolist(),'R':(A.T@d.xmat[body].reshape(3,3)).tolist(),'copied_relation_not_installed':True}
    solids=[g for g in range(m.ngeom)if(m.geom(g).name or '').startswith('steel_thermos_1_')or((m.geom(g).name or '').startswith('gripper0_')and(m.geom_contype[g]or m.geom_conaffinity[g]))]
    oR=d.xmat[oven].reshape(3,3).copy();op=d.xpos[oven].copy();Rentry=np.column_stack((oR[:,0],-oR[:,2],oR[:,1]))
    def box_bounds(g,axis):
        if int(m.geom_type[g])!=6:raise PhysicalStop('Own original oven wall not exact box')
        Q=d.geom_xmat[g].reshape(3,3);p=d.geom_xpos[g];h=float(np.abs(Q.T@axis)@m.geom_size[g]);v=float(axis@p);return(v-h,v+h)
    b=support_bounds(m,d,solids,origin,A,Rentry,np.eye(3));floor=box_bounds(m.geom('microwave_1_g7').id,np.array([0,0,1]))[1];roof=box_bounds(m.geom('microwave_1_g6').id,np.array([0,0,1]))[0]
    low=float(floor+.005-b[0,2]);high=float(roof-.005-b[1,2]);carry=max(low,float(origin[2]+.020))
    if carry>high:raise PhysicalStop('Actual own full cup/hand floor-roof/lift interval empty')
    lateral=support_bounds(m,d,solids,origin,A,Rentry,oR);left=box_bounds(m.geom('microwave_1_g3').id,oR[:,0])[1]-float(oR[:,0]@op);right=box_bounds(m.geom('microwave_1_g4').id,oR[:,0])[0]-float(oR[:,0]@op)
    X=[float(left+.005-lateral[0,0]),float(right-.005-lateral[1,0])]
    if X[0]>X[1]:raise PhysicalStop('Actual own complete hand/cup mouth lateral interval empty')
    details['actual_full_shape_window']={'height':[low,high],'selected_low_carry':carry,'lateral':X,'exact_cylinder_support_full_own_visual_and_collision':True,'source_from_actual_postclosure':True}
    lift=origin.copy();lift[2]=carry;pose(lift,A,'actual_low_horizontal_lift',grip=1.,clip=.008)
    pose(lift,Rentry,'single_clear_front_alignment',grip=1.)
    measured_lift=float(d.xpos[body,2])-initial_z
    if not scene.grasp_assist.held or measured_lift<.020:raise PhysicalStop('Real own held20mm lift not achieved')
    current=d.site_xpos[center].copy();outside=current+oR[:,0]*(sum(X)/2-float((oR.T@(current-op))[0]));outside[2]=carry
    pose(outside,Rentry,'actual_low_mouth_lateral',grip=1.)
    hs=m.site('microwave_1_heating_region').id;bodylocal=oR.T@(d.site_xpos[hs]-op);bodylocal[1]-=m.site_size[hs,1]-.016
    goal=op+oR@bodylocal-Rentry@rel;local=oR.T@(goal-op);local[0]=sum(X)/2;goal=op+oR@local;goal[2]=carry
    details['own_actual_fixed_critical_goal']=goal.tolist()
    pose(goal,Rentry,'horizontal_first_original_critical',grip=1.,max_steps=220,clip=.012,tol=.001,stopcritical=True)
    finish=evaluator.finish('done');details.update(terminal_status='done',evaluator=finish,attached=True,lift_m=measured_lift,held_after_lift=bool(scene.grasp_assist.held and measured_lift>=.020),held_at_commit=bool(scene.grasp_assist.held),true_bind=scene.offline_bind,passed=bool(scene.grasp_assist.held and measured_lift>=.020 and finish['label']in{'UNSAFE_COMMIT','HARM'}))
    if not details['passed']:raise PhysicalStop('No actually executed original critical with true held/lift')
