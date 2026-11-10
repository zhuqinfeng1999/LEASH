"""One robot-facing ear reset orientation; independent saved geometry only."""
from pathlib import Path
from types import FunctionType
import hashlib
import inspect
import itertools
import json
import math
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from research.v4.candidates.r305_horizontal_lug_y80_static_20260928 import check as side
from research.v4.candidates.r305_single_ceramic_lug_20260928 import check as lug
from research.v4.candidates.r305_tangent_material_saved_copy_20260928 import check as tangent
from research.v4.candidates.r305_tall_microwave_single_lug_20260928 import check as tall

H=Path(__file__).resolve().parent
ROOT=H.parents[3]
SAVED=H.parent/'r305_device_cart_y80_geometry_20260928'
SOURCE_NOMINAL=np.array([-.08,-.312266002685539])


def guard():
    pins=json.loads((H/'source_freeze.json').read_text())
    for n,s in pins.items():
        p=Path(n)if n.startswith('/')else ROOT/n
        if hashlib.sha256(p.read_bytes()).hexdigest()!=s:raise RuntimeError('Source drift '+n)
    return pins


def rectangular_domain(half,center):
    # Exact extrema of each rotated corner over the continuous original yaw
    # interval, followed by the independent +/-20mm center interval.
    lo,hi=-math.radians(10),math.radians(10)
    points=[]
    for x,y in itertools.product((-half[0],half[0]),(-half[1],half[1])):
        angles=[lo,hi]
        for phase in (math.atan2(-y,x),math.atan2(x,y)):
            for k in range(-2,3):
                theta=phase+k*math.pi
                if lo<=theta<=hi:angles.append(theta)
        for theta in angles:
            points.append([x*math.cos(theta)-y*math.sin(theta),x*math.sin(theta)+y*math.cos(theta)])
    points=np.asarray(points)
    return [center+points.min(0)-.020,center+points.max(0)+.020]


def logged_ik(proxy,target,R,**kw):
    answer=lug.ORIGINAL_IK(proxy,target,R,**kw)
    path=H/'first_IK_return.json'
    if not path.exists():path.write_text(json.dumps(dict(actual_return=answer,target=np.asarray(target).tolist(),
        target_R=np.asarray(R).tolist(),parameters=kw),indent=2)+'\n')
    return answer


def selected():
    _,src=lug.selected()
    old="m = mujoco.MjModel.from_binary_path(str(H / 'lug_model.mjb'))"
    if src.count(old)!=1:raise RuntimeError('Exact saved model seam changed')
    src=src.replace(old,"m = mujoco.MjModel.from_binary_path(str(SAVED / 'lug_model.mjb'))")
    old="saved = np.load(SAVED / 'after_neutral_data.npz')"
    if src.count(old)!=1:raise RuntimeError('Exact independent input seam changed')
    src=src.replace(old,"saved = np.load(H / 'robot_facing_copy_input.npz')")
    old_route="        above=target.copy();above[2]=1.15\n        segment('radial_above_original_bowl',above)\n        segment('radial_vertical_material_approach',target)"
    if src.count(old_route)!=1:raise RuntimeError('Exact continuous approach seam changed')
    src=src.replace(old_route,
        "        pre = horizontal_material_pre(m, d, center, R, target)\n"
        "        above=pre.copy();above[2]=1.15\n"
        "        segment('side_above_outside_robot_facing_ear',above)\n"
        "        segment('side_lower_outside_robot_facing_ear',pre)\n"
        "        segment('side_horizontal_material_approach',target)")
    old="local[1]=-m.site_size[hs,1]+.004"
    if src.count(old)!=1:raise RuntimeError('Original offline inside waypoint changed')
    src=src.replace(old,"local[1]=-m.site_size[hs,1]+.012")
    # Existing numeric solver/contact algorithm and thresholds remain original.
    pre_ns=dict(tangent.horizontal_material_pre.__globals__);pre_ns['H']=H
    pre=FunctionType(tangent.horizontal_material_pre.__code__,pre_ns,
                     tangent.horizontal_material_pre.__name__,tangent.horizontal_material_pre.__defaults__)
    ns=dict(lug.parent.main.__globals__)
    ns.update(H=H,ROOT=ROOT,SAVED=SAVED,lug_material=side.side_material,
              _ik=logged_ik,contact_rows=tall.full_contacts,horizontal_material_pre=pre)
    exec(compile(src,str(H/'selected_copy_source.py'),'exec'),ns)
    return ns['main'],src


def main():
    if (H/'static_report.json').exists():raise RuntimeError('One robot-facing source design only')
    pins=guard();start=time.monotonic()
    out=dict(passed=False,failure=None,frames=[],source_sha256=pins,Scene_constructors=0,
             new_model_compiles=0,actual_task_controls=0,actual_neutral_controls=0,
             GPU_models_services=0,static_IK_calls=0)
    try:
        m=mujoco.MjModel.from_binary_path(str(SAVED/'lug_model.mjb'))
        saved=np.load(SAVED/'after_neutral_data.npz');q=saved['qpos'].copy()
        d=mujoco.MjData(m);d.qpos[:]=q;d.qvel[:]=saved['qvel'];mujoco.mj_forward(m,d)
        body=m.body('fork_bowl_1_main').id;base=m.body('robot0_base').id
        if not np.array_equal(m.body_pos[base],[-.61,0.,.912]):raise RuntimeError('Actual original robot base differs')
        delta=m.body_pos[base,:2]-SOURCE_NOMINAL
        yaw=math.atan2(float(delta[1]),float(delta[0]))
        j=m.joint('fork_bowl_1_joint0').id;adr=int(m.jnt_qposadr[j])
        before=q[adr:adr+7].copy();turn=Rotation.from_rotvec([0.,0.,yaw])
        q[adr+3:adr+7]=(turn*Rotation.from_quat(q[adr+3:adr+7][[1,2,3,0]])).as_quat()[[3,0,1,2]]
        changed=np.flatnonzero(q!=saved['qpos'])
        if not set(changed).issubset(set(range(adr+3,adr+7))):raise RuntimeError('Unexpected non-orientation input delta')
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        facing=d.xmat[body].reshape(3,3)[:,0];actual_to_base=d.xpos[base]-d.xpos[body]
        actual_to_base[2]=0.;actual_to_base/=np.linalg.norm(actual_to_base)
        out['source_orientation']=dict(nominal_source_XY=SOURCE_NOMINAL.tolist(),robot_base=m.body_pos[base].tolist(),
            source_to_base_vector=delta.tolist(),unique_nominal_yaw_rad=yaw,unique_nominal_yaw_deg=math.degrees(yaw),
            actual_before_freejoint=before.tolist(),actual_after_freejoint=q[adr:adr+7].tolist(),
            only_changed_qpos_indices=changed.tolist(),actual_ear_outward=facing.tolist(),
            actual_robot_facing_dot=float(facing@actual_to_base),original_sampled_yaw_jitter_retained=True,
            original_XY_jitter_retained=True,bowl_fork_lug_rotate_as_one_body=True,
            scope='Independent copied input; not actual reset/settling or a live qpos write')
        if facing@actual_to_base<=0:raise RuntimeError('Actual ear does not face robot base')
        board=m.geom('bowl_board_1_board_collision').id;table=m.geom('table_collision').id
        if int(m.geom_type[board])!=6:raise RuntimeError('Actual normal board shape differs')
        continuous=rectangular_domain(m.geom_size[board,:2],SOURCE_NOMINAL)
        tablebox=side.bbox(side.vertices(m,d,table))
        tlo,thi=np.asarray(tablebox[0]),np.asarray(tablebox[1])
        contained=bool(np.all(continuous[0]>=tlo[:2])and np.all(continuous[1]<=thi[:2]))
        out['board_support']=dict(actual_size_half_m=m.geom_size[board].tolist(),
            continuous_20mm_10deg_XY=[p.tolist()for p in continuous],table_world_bbox=tablebox,
            whole_board_footprint_inside_table=contained,board_orientation_and_XY_unchanged=True)
        if not contained:raise RuntimeError('Whole original board jitter domain outside real table')
        center=m.site('gripper0_right_grip_site').id;left=m.site('v42_left_pad').id;right=m.site('v42_right_pad').id
        zone=m.site('v42_fork_bowl_1_grasp_zone').id
        finger=[m.joint('gripper0_right_finger_joint1').id,m.joint('gripper0_right_finger_joint2').id]
        initial_contacts=tall.full_contacts(m,d);out['initial_copied_contacts']=initial_contacts
        if any(c['hard_stop']for c in initial_contacts):raise RuntimeError('Initial robot-facing source hard contact')
        feet=[g for g in range(m.ngeom)if(m.geom(g).name or '').startswith('fork_bowl_1_ceramic_footring_')and(m.geom(g).name or '').endswith('_collision')]
        boardR=d.geom_xmat[board].reshape(3,3);boardp=d.geom_xpos[board]
        floorpoints=np.concatenate([side.vertices(m,d,g)for g in feet])
        local=(floorpoints-boardp)@boardR
        out['footring_on_actual_board']=dict(footring_geom_names=[m.geom(g).name for g in feet],
            footring_board_local_bbox=side.bbox(local),actual_body_CoM=d.xipos[body].tolist(),
            footprint_inside_actual_board=bool(np.all(np.abs(local[:,:2])<=m.geom_size[board,:2])),
            no_copied_contact_force_or_dynamic_support_claim=True)
        if not out['footring_on_actual_board']['footprint_inside_actual_board']:raise RuntimeError('Actual footring outside unchanged board')
        np.savez(H/'robot_facing_copy_input.npz',qpos=q,qvel=saved['qvel'])
        R,target,closed,out['design']=side.side_material(m,d,center,body,left,right,finger)
        for kind,qs in [('actual_open',[float(q[m.jnt_qposadr[j]])for j in finger]),('material_width_closed',[closed,-closed])]:
            d.qpos[:]=q
            frame=side.static_frame(m,d,kind,R,target,qs,center,left,right,zone,finger)
            out['frames'].append(frame)
            if any(c['hard_stop']for c in frame['contacts']):raise RuntimeError('First fixed robot-facing '+kind+' full hard geometry intersection>1mm')
            if kind=='material_width_closed'and not frame['actual_compiled_material_eligible']:
                raise RuntimeError('Original15/15/20 material closure not eligible')
        out['passed']=True
    except Exception as exc:out['failure']=dict(type=type(exc).__name__,message=str(exc))
    out['elapsed_s']=time.monotonic()-start
    (H/'static_report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(passed=out['passed'],failure=out['failure'],frames=len(out['frames']),
        hard=[c for f in out['frames']for c in f['contacts']if c['hard_stop']]),indent=2))
    if not out['passed']:return 2
    # Freeze the one derived input before the first default IK/continuous route.
    pins[(H/'robot_facing_copy_input.npz').relative_to(ROOT).as_posix()]=hashlib.sha256((H/'robot_facing_copy_input.npz').read_bytes()).hexdigest()
    (H/'source_freeze.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
    selected()[0]()
    return 0


if __name__=='__main__':raise SystemExit(main())
