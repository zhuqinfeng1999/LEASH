"""One original top material frame/route on the saved unraised own model."""
from pathlib import Path
from types import FunctionType
import hashlib
import itertools
import json
import math

import mujoco
import numpy as np

from research.v4.candidates.r305_single_ceramic_lug_20260928 import check as original
from research.v4.candidates.r305_side_jaw_mirror_seed2_copy_20260928 import check as logger

H=Path(__file__).resolve().parent
ROOT=H.parents[3]
MODEL=H.parent/'r305_robot_facing_lug_y80_copy_20260928/actual_constructor_seed1/actual_model.mjb'
INPUT=H.parent/'r305_side_jaw_mirror_seed2_copy_20260928/robot_facing_copy_input.npz'


def endpoint_probe(m,d,proxy,center,qidx,R,target,result,logged):
    answer=logged(proxy,target,R,max_iter=200,position_tol=.012,rotation_tol=.03)
    result['original_default_position_endpoint_probe']=dict(actual_return=answer,
        original_parameters=dict(max_iter=200,position_tol=.012,rotation_tol=.03),
        no_newsolver_or_seed_search=True,scope='Independent default-position endpoint prerequisite, not closure or route')
    if not answer['reachable']or answer.get('restart',0)!=0:
        raise RuntimeError('First standard top default endpoint IK unavailable under original continuity guard')
    d.qpos[qidx]=answer['joint_qpos_rad'];mujoco.mj_forward(m,d)
    contacts=logger.full_contacts(m,d)
    result['original_default_position_endpoint_probe'].update(qpos=d.qpos.tolist(),
        actual_EEF=d.site_xpos[center].tolist(),actual_R=d.site_xmat[center].reshape(3,3).tolist(),
        joint_limits=logger.joint_rows(m,d),contacts=contacts)
    if any(c['hard_stop']for c in contacts):
        raise RuntimeError('First standard top default endpoint open-hand hard collision>1mm')


def clearance_waypoint(m,d,center,R,target,result):
    ep=d.site_xpos[center].copy();E=d.site_xmat[center].reshape(3,3)
    points=[];source=[]
    for g in range(m.ngeom):
        name=m.geom(g).name or ''
        if not(m.geom_contype[g]or m.geom_conaffinity[g]):continue
        a=m.geom_aabb[g];G=d.geom_xmat[g].reshape(3,3)
        v=np.array([d.geom_xpos[g]+G@(a[:3]+a[3:]*np.asarray(s))for s in itertools.product((-1.,1.),repeat=3)])
        if name.startswith('gripper0_')or name=='robot0_link7_collision':
            points.extend(((v-ep)@E)@R.T)
        elif name.startswith(('fork_bowl_1_','bowl_board_1_')):
            source.extend(v)
    hand=np.asarray(points);props=np.asarray(source)
    min_clear=float(props[:,2].max()+.005+.012-hand[:,2].min())
    above=target.copy();above[2]=ep[2]
    result['source_clearance_waypoint']=dict(EEF=above.tolist(),
        actual_existing_home_EEF=ep.tolist(),minimum_whole_shape_clearance_EEF_z_m=min_clear,
        highest_source_and_board_m=float(props[:,2].max()),lowest_hand_wrist_relative_z_m=float(hand[:,2].min()),
        original_approach_residual_m=.012,existing_geometry_clearance_m=.005,
        actual_clearance_beyond_12mm_residual_m=float(above[2]+hand[:,2].min()-props[:,2].max()-.012),
        reused_actual_home_height=True,no_new_height_grid=True,
        bounds_scope='Conservative transformed compiled whole geom_aabb, including capsules/cylinders; full contact STOP remains actual shapes')
    if above[2]<min_clear:raise RuntimeError('Existing home height cannot clear complete source with original approach residual')
    return ep,above


def selected():
    callback,source=original.selected()
    substitutions={
        "m = mujoco.MjModel.from_binary_path(str(H / 'lug_model.mjb'))":'m = mujoco.MjModel.from_binary_path(str(MODEL))',
        "saved = np.load(SAVED / 'after_neutral_data.npz')":'saved = np.load(INPUT)',
        'local[1]=-m.site_size[hs,1]+.004':'local[1]=-m.site_size[hs,1]+.012',
        'ik=_ik(proxy,p,Q,max_iter=200,position_tol=.008,rotation_tol=.03)':
        "ik=_ik(proxy,p,Q,max_iter=200,position_tol=(.001 if phase=='standard_top_vertical_material_approach' and np.array_equal(p,target) else .012),rotation_tol=(math.radians(1) if phase=='standard_top_vertical_material_approach' and np.array_equal(p,target) else .03))",
        "        above=target.copy();above[2]=1.15\n        segment('radial_above_original_bowl',above)\n        segment('radial_vertical_material_approach',target)":
        "        home,above=clearance_waypoint(m,d,center,R,target,result)\n        segment('standard_top_orient_at_existing_home',home)\n        segment('standard_top_lateral_above_material',above)\n        segment('standard_top_vertical_material_approach',target)",
        'carry=max(float(op[2]+lower),float(grip[2]+.020))':'carry=max(float(op[2]+lower),float(grip[2]+.032))',
        "        lift=grip.copy();lift[2]=carry;segment('radial_source_low_lift',lift)":
        "        lift=grip.copy();lift[2]=carry;segment('radial_source_low_lift',lift)\n"
        "        result['copied_geometric_lift']={'requested_source_lift_waypoint_m':.032,'actual_frozen_lift_minimum_m':.020,\n"
        "            'payload_Z_delta_m':float(d.xpos[body,2]-saved_source_z),'no_actual_bind_lift_claim':True}\n"
        "        if d.xpos[body,2]-saved_source_z<.020:raise RuntimeError('Copied payload did not reach original20mm lift geometry')",
        "        initial = d.qpos.copy()":"        initial = d.qpos.copy()\n        saved_source_z=float(d.xpos[m.body('fork_bowl_1_main').id,2])",
    }
    for before,after in substitutions.items():
        if source.count(before)!=1:raise RuntimeError('Exact original top route seam differs: '+before)
        source=source.replace(before,after)
    a="        ik = _ik(proxy,target,R,max_iter=200,position_tol=.001,rotation_tol=.03)\n"
    z="        result['static_two_frames_passed']=True\n"
    if source.count(a)!=1 or source.count(z)!=1:raise RuntimeError('Original endpoint/static boundary differs')
    start,end=source.index(a),source.index(z)+len(z)
    source=source[:start]+(
        "        endpoint_probe(m,d,proxy,center,qidx,R,target,result,_ik)\n"
        "        helper=GraspAssist.__new__(GraspAssist);helper.model=m;helper.data=d\n"
        "        helper._center=center;helper._left=left;helper._right=right\n"
    )+source[end:]
    log=FunctionType(logger.logged_ik.__code__,logger.logged_ik.__globals__|dict(H=H,IK_LOG=[]),
        logger.logged_ik.__name__,logger.logged_ik.__defaults__,logger.logged_ik.__closure__)
    namespace=callback.__globals__|dict(H=H,ROOT=ROOT,MODEL=MODEL,INPUT=INPUT,
        _ik=log,contact_rows=logger.full_contacts,math=math,
        endpoint_probe=endpoint_probe,clearance_waypoint=clearance_waypoint)
    exec(compile(source,str(H/'selected_top_source.py'),'exec'),namespace)
    fn=namespace['main']
    if fn.__globals__['lug_material']is not original.lug_material:
        raise RuntimeError('Original standard top material helper was replaced')
    if log.__globals__['ORIGINAL_IK']is not original.ORIGINAL_IK:
        raise RuntimeError('Original numerical solver replaced')
    return fn,source


def prepare():
    if(H/'selected_copy_source.txt').exists():raise RuntimeError('Selection is write-once')
    _,source=selected();(H/'selected_copy_source.txt').write_text(source)
    (H/'selection_contract.json').write_text(json.dumps(dict(passed=True,
        original_material_helper_identity=True,original_numerical_solver_identity=True,
        original_full_contact_and_pad_only_exception=True,
        saved_model=str(MODEL.relative_to(ROOT)),saved_input=str(INPUT.relative_to(ROOT)),
        no_pedestal_or_side_pre=True,only_requested_normal_top_sequence=True,
        default_endpoint_and_intermediate=dict(max_iter=200,restarts=[0,1],position_tol=.012,rotation_tol=.03),
        original_continuity_requires_restart_zero=True,
        final_material=dict(position_tol=.001,rotation_tol_deg=1),requested_lift_waypoint_m=.032,
        original_lift_rule_m=.020,original_inside_private_waypoint_m=.012,
        original_budget=dict(total_controls=1200,wall_seconds=90,common_reset_controls=20),
        Scene_compile_IK_taskcontrols_GPU=0,no_actualexpert_or_factory_created=True),indent=2)+'\n')


def main():
    fn,source=selected()
    if source!=(H/'selected_copy_source.txt').read_text():raise RuntimeError('Frozen selected source differs')
    fn()


if __name__=='__main__':main()
