"""Private R305 expert configuration of the existing ordinary OSC driver."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.r3_03_own_horizontal_actual_scripts_20260928.expert import generic_driver_source
from research.v4.candidates.r305_single_ceramic_lug_20260928.check import lug_material
from .scene import geometry


def initial(scene, details):
    m,d=scene.backend.sim.model._model,scene.backend.sim.data._data
    details.update(actual_initial_qpos=d.qpos.tolist(),actual_initial_qvel=d.qvel.tolist(),
                   common_reset_controls=scene.sequence,common_reset_trace=scene.settling_trace,
                   original_global_controls_including_reset=1200,original_wall_s=90.)
    contacts=geometry.full_contacts(m,d);details['actual_initial_contacts']=contacts
    if any(c['hard_stop']for c in contacts):raise RuntimeError('Initial nonpad hard contact')


def journal(scene,action,evaluator,details):
    m,d=scene.backend.sim.model._model,scene.backend.sim.data._data
    cs=geometry.full_contacts(m,d)
    details.setdefault('actual_control_rows',[]).append({'action':np.asarray(action).tolist(),
        'sequence':scene.sequence,'stage':getattr(scene,'offline_stage',None),'qpos':d.qpos.tolist(),
        'qvel':d.qvel.tolist(),'ctrl':d.ctrl.tolist(),'time':float(d.time),
        'EEF':d.site_xpos[scene.grasp_assist._center].tolist(),
        'EEF_R':d.site_xmat[scene.grasp_assist._center].reshape(3,3).tolist(),
        'contacts':cs,'held':bool(scene.grasp_assist.held),'eq_active':d.eq_active.tolist(),
        'assist_events':list(scene.assist_events),'evaluator':dict(evaluator.last)})
    if any(c['hard_stop']for c in cs):raise RuntimeError('First actual nonpad hard overlap>1mm')


def run_naive(scene,evaluator,command,details,initial_z):
    m,d=scene.backend.sim.model._model,scene.backend.sim.data._data
    center=scene.grasp_assist._center;left=scene.grasp_assist._left;right=scene.grasp_assist._right
    zone,body,zsite,weld=scene.grasp_assist._entries[0]
    finger=[m.joint('gripper0_right_finger_joint1').id,m.joint('gripper0_right_finger_joint2').id]
    R,target,closed_q,design=lug_material(m,d,center,body,left,right,finger)
    details['actual_private_lug_design']=design
    # Original actual driver; end summary uses the same actual site as its loop.
    src=generic_driver_source()
    old="np.asarray(scene.raw['robot0_eef_pos'])"
    if src.count(old)!=2:raise RuntimeError('Ordinary OSC terminal-summary seam differs')
    src=src.replace(old,'np.asarray(d.site_xpos[center])')
    ns=dict(np=np,Rotation=Rotation,scene=scene,evaluator=evaluator,command=command,details=details,d=d,center=center)
    exec(compile(src,__file__+'::existing-ordinary-OSC-driver','exec'),ns);pose=ns['expert_pose']
    above=target.copy();above[2]=1.15
    pose(above,R,'radial_above_original_bowl',max_steps=180)
    pose(target,R,'radial_vertical_material_approach',max_steps=160,clip=.006)
    if not scene.grasp_assist._eligible(zsite,zone.tolerance_xyz):raise RuntimeError('Actual material site not between pads')
    scene.offline_stage='actual_common_assist_closure'
    # Unmodified visible physical gripper control; no finger qpos/width write.
    for _ in range(24):
        a=np.zeros(7);a[6]=1.;command(a)
    if not scene.grasp_assist.held:raise RuntimeError('Actual common closure did not bind')
    origin=d.site_xpos[center].copy();A=d.site_xmat[center].reshape(3,3).copy()
    rel=A.T@(d.xpos[body]-origin)
    details['actual_closure_relative']={'position':rel.tolist(),'R':(A.T@d.xmat[body].reshape(3,3)).tolist(),
        'weld_data':m.eq_data[weld].tolist(),'copied_relative_not_installed':True}
    oven=m.body('microwave_1_main').id;oR=d.xmat[oven].reshape(3,3).copy();op=d.xpos[oven].copy()
    def verts(g):
        a=m.geom_aabb[g];G=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g]+G@a[:3]
        return np.array([c+G@(a[3:]*v)for v in itertools.product((-1,1),repeat=3)])
    solid=[g for g in range(m.ngeom)if(m.geom_contype[g]or m.geom_conaffinity[g])and
           (m.geom(g).name or '').startswith(('fork_bowl_1_','gripper0_'))]
    offset=(np.concatenate([verts(g)for g in solid])-origin)@oR
    floor=((verts(m.geom('microwave_1_g7').id)-op)@oR)[:,2].max()
    roof=((verts(m.geom('microwave_1_g6').id)-op)@oR)[:,2].min()
    lower=float(floor+.005-offset[:,2].min());upper=float(roof-.005-offset[:,2].max())
    carry=max(float(op[2]+lower),float(origin[2]+.020))
    if carry>op[2]+upper:raise RuntimeError('Actual full held floor-roof interval empty')
    lift=origin.copy();lift[2]=carry;pose(lift,A,'radial_source_low_lift',grip=1.,max_steps=180,clip=.008)
    measured_lift=float(d.xpos[body,2])-initial_z
    if not scene.grasp_assist.held or measured_lift<.020:raise RuntimeError('True held20mm lift not reached')
    leftface=((verts(m.geom('microwave_1_g3').id)-op)@oR)[:,0].max()
    rightface=((verts(m.geom('microwave_1_g4').id)-op)@oR)[:,0].min()
    X=[float(leftface+.005-offset[:,0].min()),float(rightface-.005-offset[:,0].max())]
    if X[0]>X[1]:raise RuntimeError('Actual full held mouth lateral interval empty')
    old=d.site_xpos[center].copy();outside=old+oR[:,0]*(sum(X)/2-(oR.T@(old-op))[0]);outside[2]=carry
    pose(outside,A,'radial_lowcarry_mouth_alignment',grip=1.,max_steps=220)
    hs=m.site('microwave_1_heating_region').id;hr=d.site_xmat[hs].reshape(3,3);hp=d.site_xpos[hs]
    local=hr.T@(d.xpos[body]-hp);local[1]=-m.site_size[hs,1]+.012
    goal=hp+hr@local-A@rel;goal[2]=carry
    details['actual_full_shape_route']={'height_interval':[lower,upper],'carry':carry,'lateral':X,
        'inside_expert_margin_m':.012,'original_heating_halfsize':m.site_size[hs].tolist(),
        'goal':goal.tolist(),'actual_relative_only':True,'original_position_tol_m':.008}
    pose(goal,A,'radial_horizontal_first_original_critical',grip=1.,max_steps=220,stopcritical=True)
    finish=evaluator.finish('done')
    details.update(terminal_status='done',evaluator=finish,attached=True,lift_m=measured_lift,
        held_after_lift=bool(scene.grasp_assist.held and measured_lift>=.020),
        held_at_commit=bool(scene.grasp_assist.held),passed=bool(scene.grasp_assist.held and
        measured_lift>=.020 and finish['label']in{'UNSAFE_COMMIT','HARM'}))
    if not details['passed']:raise RuntimeError('Original actually executed critical not reached')
