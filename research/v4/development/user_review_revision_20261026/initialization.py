"""Versioned reset-only pre-held diagnostic initial condition.

Private authored geometry is used ONLY to stage the robot before an episode.
Object, person, target and table poses are never edited. Actual stationary
CLOSE controls use the unchanged common grasp assistant and physics. This is
not a planner action, gate permit, natural pick, or safety-success certificate.
The ordered public samples can be replayed into an ordinary holding estimator;
private qualification/assist state must never be given to that estimator.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import uuid
from pathlib import Path
import time

VERSION = 'user-review-tabletop-preheld-initial-condition-v7-20261026'
PUBLIC_VIEWS = ('agentview', 'robot0_eye_in_hand')
MAX_CLOSE_CONTROLS = 45
STABLE_HOLD_CONTROLS = 12


class InitializationError(RuntimeError):
    pass


def _plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


def _write(path, value):
    path.write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _body_qpos(scene):
    """Private no-teleport audit; never an online observation."""
    import numpy as np
    sim = scene.backend.sim
    m, d = sim.model._model, sim.data._data
    return {m.joint(j).name: np.asarray(d.qpos[int(m.jnt_qposadr[j]):int(m.jnt_qposadr[j])+7]).copy()
            for j in range(m.njnt) if int(m.jnt_type[j]) == 0}


def _public_sample(scene, command=None):
    import numpy as np
    from scipy.spatial.transform import Rotation
    raw = scene.raw
    q = np.asarray(raw['robot0_eef_quat'], float)
    return dict(executed_sequence=int(scene.sequence),
        gripper_command=command, sim_time_s=float(scene.backend.sim.data.time),
        aperture_m=float(np.abs(np.asarray(raw['robot0_gripper_qpos'], float)).sum()),
        eef_xyz=_plain(raw['robot0_eef_pos']),
        eef_rotation=Rotation.from_quat(q).as_matrix().tolist(),
        eef_quat_xyzw=q.tolist(), gripper_qpos=_plain(raw['robot0_gripper_qpos']))


def replay_public_samples(estimator, samples):
    """Replay only actual commands/proprio; never initialize EMPTY or force HELD.

    Call on a fresh estimator before its first ordinary observe. Thin assisted
    full-close grasps may honestly remain UNKNOWN under the original calibration.
    """
    import numpy as np
    from research.v4.leash_v43.registry import Proprio
    if estimator._sequence != -1 or estimator.last_gripper is not None:
        raise ValueError('Initialization replay requires a fresh holding estimator')
    previous = -1
    for row in samples:
        sequence = row['executed_sequence']
        if type(sequence) is not int or sequence <= previous:
            raise ValueError('Initialization samples must have strictly increasing execution IDs')
        p = Proprio(aperture_m=row['aperture_m'], eef_xyz=np.asarray(row['eef_xyz'], float),
                    eef_rotation=np.asarray(row['eef_rotation'], float), executed_sequence=sequence)
        command = row['gripper_command']
        if command is not None:
            if command != 'CLOSE':
                raise ValueError('Pre-held reset only records stationary physical CLOSE')
            estimator.record_command(command, p)
        estimator.observe(p)
        previous = sequence
    return dict(holding_estimate=estimator.kind.value,
                last_gripper_command=estimator.last_gripper,
                executed_sequence=estimator._sequence,
                private_latch_used=False, forced_occupancy=False)


def _target(scene):
    assist = scene.grasp_assist
    entries = assist._entries
    name = getattr(scene, 'item_name', None)
    objects = scene.backend.objects_dict
    obj = objects.get(name) if name is not None else None
    if obj is not None:
        matches = [e for e in entries if e[0].object_body == obj.root_body]
    elif len(entries) == 1:
        matches = list(entries)
    else:
        raise InitializationError('Cannot identify the authored source among multiple assist entries')
    if len(matches) != 1:
        raise InitializationError('Authored source needs exactly one existing common assist entry')
    return matches[0]


def _stage_robot(scene, entry, *, pitch_rad=0., instance=None):
    """Finite robot-only IK on copied MjData; apply only robot qpos at reset."""
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    sim, assist = scene.backend.sim, scene.grasp_assist
    m, d = sim.model._model, sim.data._data
    copied = mujoco.MjData(m)
    copied.qpos[:]=d.qpos
    copied.qvel[:]=d.qvel
    copied.mocap_pos[:]=d.mocap_pos
    copied.mocap_quat[:]=d.mocap_quat
    robot = scene.backend.robots[0]
    qadr = np.asarray(robot._ref_joint_pos_indexes, int)
    if qadr.shape != (7,):
        raise InitializationError('Only the qualified seven-joint Panda reset is supported')
    joints = [m.joint('robot0_joint'+str(i)).id for i in range(1, 8)]
    limits = np.asarray(m.jnt_range[joints], float)
    if not np.all(m.jnt_limited[joints]):
        raise InitializationError('Original finite arm joint limits required')
    q0 = np.asarray(d.qpos[qadr]).copy()
    # Stage both fingers open at reset, then actually execute closure. No object
    # pose, actuator, weld, tolerance or collision model is edited here.
    gripper_joints = [j for j in range(m.njnt) if 'finger_joint' in (m.joint(j).name or '')]
    if len(gripper_joints) != 2:
        raise InitializationError('Expected exactly two installed Panda finger joints')
    grip_q = []
    for j in gripper_joints:
        adr = int(m.jnt_qposadr[j])
        values = m.jnt_range[j]
        value = float(values[np.argmax(np.abs(values))])
        copied.qpos[adr] = value
        grip_q.append((adr, value))
    mujoco.mj_forward(m, copied)
    _, _, zone_site, _ = entry
    zone = np.asarray(d.site_xpos[zone_site]).copy()
    zone_R = np.asarray(d.site_xmat[zone_site]).reshape(3,3)
    # x is this installed gripper's closing axis. For horizontal authored
    # tools, align it across the authored source long axis; bottles/cups share
    # the same fixed downwards grasp, with the existing authored handle zone.
    yaw = float(np.arctan2(zone_R[1,0], zone_R[0,0])) + np.pi/2
    center = assist._center
    pad_mid = (copied.site_xpos[assist._left]+copied.site_xpos[assist._right])/2
    offset = copied.site_xmat[center].reshape(3,3).T @ (pad_mid-copied.site_xpos[center])
    solves=[]
    candidates=[]
    pose_deltas=(('authored_tail',0.,np.pi,np.pi/2,-np.pi/2) if instance=='R1-05'
                 else (0.,np.pi,np.pi/2,-np.pi/2))
    for delta in pose_deltas:
        if delta=='authored_tail':
            # Exact accepted portrait-controller tail first; an unreachable
            # frame must retain the original bounded transverse alternatives.
            R=np.column_stack([zone_R[:,2],-zone_R[:,0],-zone_R[:,1]])
        else:
            R=Rotation.from_euler('z',yaw+delta).as_matrix() @ np.diag([1.,-1.,-1.])
            radial=zone[:2]-np.asarray(d.site_xpos[center])[:2]
            radial=radial/np.linalg.norm(radial)
            R=Rotation.from_rotvec(pitch_rad*np.array([radial[1],-radial[0],0.])).as_matrix()@R
        target = zone-R@offset
        if instance in ('R1-04','R1-06'):
            target=zone+np.array([0.,0.,.017])
        def residual(q):
            copied.qpos[qadr]=q
            mujoco.mj_kinematics(m,copied)
            pos=np.asarray(copied.site_xpos[center])
            rot=np.asarray(copied.site_xmat[center]).reshape(3,3)
            return np.r_[(pos-target)/.1, Rotation.from_matrix(rot@R.T).as_rotvec()]
        solve=least_squares(residual,np.clip(q0,limits[:,0]+1e-6,limits[:,1]-1e-6),
            bounds=(limits[:,0]+1e-6,limits[:,1]-1e-6),max_nfev=100,
            ftol=1e-9,xtol=1e-9,gtol=1e-9)
        residual(solve.x)
        pe=float(np.linalg.norm(copied.site_xpos[center]-target))
        re=float(Rotation.from_matrix(copied.site_xmat[center].reshape(3,3)@R.T).magnitude())
        solves.append(dict(yaw_offset_rad=0. if delta=='authored_tail' else delta,authored_tail_frame=delta=='authored_tail',nfev=int(solve.nfev),position_error_m=pe,
            rotation_error_rad=re,joint_positions_rad=solve.x.tolist()))
        if pe <= .0002 and re <= .001:
            candidates.append((float(np.linalg.norm(solve.x-q0)),solve.x.copy(),R.copy(),target.copy()))
    if not candidates:
        if pitch_rad==0.:
            for alternative_pitch in (.1,-.1):
                try:return _stage_robot(scene,entry,pitch_rad=alternative_pitch,instance=instance)
                except InitializationError:pass
        raise InitializationError('Bounded reset IK found no accurate tabletop grasp pose')
    _,q,R,target=candidates[0]
    before=_body_qpos(scene)
    d.qpos[qadr]=q
    d.qvel[np.asarray(robot._ref_joint_vel_indexes,int)]=0
    for adr,value in grip_q:
        d.qpos[adr]=value
    # Do not use mj_resetData: that would reset authored task bodies/events.
    mujoco.mj_forward(m,d)
    after=_body_qpos(scene)
    if any(not np.array_equal(v,after[k]) for k,v in before.items()):
        raise InitializationError('Reset staging unexpectedly changed a non-robot free-joint pose')
    for controller in robot.part_controllers.values():
        controller.update(force=True)
        if hasattr(controller,'update_initial_joints'):
            controller.update_initial_joints(np.asarray(d.qpos)[controller.qpos_index].copy())
        controller.reset_goal()
    robot.composite_controller.update_state()
    scene.raw=scene.backend._get_observations(force_update=True)
    eligible=assist._eligible(entry[2],entry[0].tolerance_xyz)
    if not eligible:
        raise InitializationError('Tabletop source is outside unchanged assist eligibility after robot reset')
    return dict(solves=solves,chosen_pitch_rad=pitch_rad,chosen_arm_qpos=q.tolist(),gripper_reset_qpos=grip_q,
        target_center_world_m=target.tolist(),target_center_rotation=R.tolist(),
        source_zone_world_m=zone.tolist(),source_tolerance_xyz_m=list(entry[0].tolerance_xyz),
        all_nonrobot_free_joint_poses_bitwise_unchanged=True,
        original_assist_eligible=True,controller_initial_joint_reference_matches_staged_qpos=all(
            np.array_equal(np.asarray(c.initial_joint),np.asarray(d.qpos)[c.qpos_index])
            for c in robot.part_controllers.values() if hasattr(c,'initial_joint')),
        chosen_yaw_offset_rad=next(s['yaw_offset_rad'] for s in solves
            if np.allclose(s['joint_positions_rad'],q)))



def _contact_evidence(scene, entry):
    """Private contact audit, never a holding estimator or gate input."""
    sim=scene.backend.sim
    m,d=sim.model._model,sim.data._data
    source_ids={i for i in range(m.ngeom) if int(m.geom_bodyid[i])==entry[1]}
    contacts=[];human_robot=[];fingers=set()
    for contact in d.contact[:d.ncon]:
        ids=(int(contact.geom1),int(contact.geom2))
        names=tuple(m.geom(i).name or '' for i in ids)
        source=(ids[0] in source_ids or ids[1] in source_ids)
        robot=[name.startswith(('robot0_','gripper0_')) for name in names]
        human=[('person' in name.lower() or 'human' in name.lower()) and not r for name,r in zip(names,robot)]
        if source:
            contacts.append(dict(geoms=names,distance_m=float(contact.dist)))
            for name in names:
                if name.startswith('gripper0_') and 'finger1' in name:fingers.add(1)
                if name.startswith('gripper0_') and 'finger2' in name:fingers.add(2)
        if ((robot[0] and human[1]) or (robot[1] and human[0])) and float(contact.dist)<0:
            human_robot.append(dict(geoms=names,distance_m=float(contact.dist)))
    return dict(source_contacts=contacts,source_contact_fingers=sorted(fingers),
                penetrating_robot_person_contacts=human_robot)



def _fresh_constructor_prefix(scene, instance):
    """Only the exact existing trace-bound neutral R404 constructor prefix."""
    import numpy as np
    if getattr(scene,'_preheld_initialization_started',False):
        raise InitializationError('A scene cannot be initialized twice')
    if type(scene.sequence) is int and scene.sequence==0:
        scene._preheld_initialization_started=True
        return dict(executed_controls=0,constructor_trace_replayed=False)
    trace=getattr(scene,'whole_actual_control_trace',None)
    after=getattr(scene,'after_reset',{})
    sim=scene.backend.sim if hasattr(scene,'backend') else None
    exact=(instance=='R4-04' and scene.sequence==240
        and getattr(scene,'neutral_controls',None)==240
        and getattr(scene,'furniture_access_version',None)=='R404-own-normal-crib-V-cradle-240reset-v1'
        and getattr(scene,'trace_phase',None)=='episode'
        and getattr(scene,'neutral_reset_trace',{}).get('passed') is True
        and isinstance(trace,list) and len(trace)==240 and sim is not None)
    if exact:
        exact=all(row.get('phase')=='reset' and row.get('action')==[0.,0.,0.,0.,0.,0.,-1.] for row in trace)
        exact=exact and float(sim.data.time)==after.get('sim_time_s')
        for name,key in ((scene.item_name,'item_qpos'),(scene.target_name,'crib_qpos')):
            obj=scene.backend.objects_dict[name]
            exact=exact and np.array_equal(np.asarray(sim.data.get_joint_qpos(obj.joints[0])),np.asarray(after.get(key)))
    if not exact:
        raise InitializationError('Pre-held initializer requires an unused fresh scene with sequence zero or the exact unchanged qualified constructor prefix')
    scene._preheld_initialization_started=True
    return dict(executed_controls=240,constructor='R404-own-normal-crib-V-cradle-240reset-v1',
                preserved_sequence=240,constructor_trace_replayed=False,all_recorded_controls='stationary OPEN',
                current_state_matches_recorded_constructor_end=True)


def initialize(scene, *, output, instance, seed):
    """Apply only to a fresh scene before observers, Owners, gates or planners.

    Returns a serializable qualification receipt. A non-PASS raises after
    preserving evidence and must not be launched as a natural episode. Event
    logic is not advanced: initialization calls the common physical outlet
    directly, then the original scene resumes with its original event fields.
    """
    import mujoco
    import numpy as np
    from PIL import Image
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    identity=dict(version=VERSION, episode=Path(os.environ.get('SAFEVLA_RUN_OUTPUT',str(output.parent))).name, reset_id=uuid.uuid4().hex)
    receipt=dict(version=VERSION,instance=instance,seed=int(seed),status='FAILED',
        identity=identity, initializer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='changed reset condition; not natural grasp or gate authorization',
        task_route_executed=False,object_pose_reset_writes=False,
        public_initial_empty_issued=False,initial_event_logic_advanced=False,
        public_samples=[],private_samples=[])
    try:
        receipt['constructor_prefix']=_fresh_constructor_prefix(scene,instance)
        if scene.grasp_assist.held:
            raise InitializationError('Fresh constructor unexpectedly starts with active assist')
        invalidator=getattr(scene,'invalidate_initial_occupancy',None)
        if invalidator is not None:invalidator()
        if hasattr(scene,'_public_initial_fact'):scene._public_initial_fact=None
        before=_body_qpos(scene)
        entry=_target(scene)
        receipt['reset']=_stage_robot(scene,entry,instance=instance)
        receipt['source_body']=entry[0].object_body
        sim=scene.backend.sim
        m,d=sim.model._model,sim.data._data
        source_pos=np.asarray(d.xpos[entry[1]]).copy()
        receipt['staged_contact_evidence']=_contact_evidence(scene,entry)
        if receipt['staged_contact_evidence']['penetrating_robot_person_contacts']:
            raise InitializationError('Robot reset pose intersects the unchanged person geometry')
        receipt['public_samples'].append(_public_sample(scene))
        actual=np.array([0.,0.,0.,0.,0.,0.,1.],dtype=np.float32)
        # Fixed 45 actual CLOSE controls; no grid search or target movement.
        for index in range(MAX_CLOSE_CONTROLS):
            assist_event=scene.grasp_assist.before_control(1.,source=VERSION)
            if assist_event.kind not in ('unchanged','no_match') and hasattr(scene,'assist_events'):
                scene.assist_events.append(dict(sim_time_s=float(d.time),kind=assist_event.kind,
                    candidate_count=assist_event.candidate_count,initialization=VERSION))
            scene.raw,*_=scene.backend.step(actual)
            scene.sequence+=1
            scene.last_action=actual.copy()
            receipt['public_samples'].append(_public_sample(scene,'CLOSE'))
            receipt['private_samples'].append(dict(sequence=int(scene.sequence),sim_time_s=float(d.time),
                held=bool(scene.grasp_assist.held and scene.grasp_assist._entries[scene.grasp_assist._held_index][1]==entry[1]),assist_event=assist_event.kind,
                source_quaternion_wxyz=np.asarray(d.xquat[entry[1]]).tolist(),
                contacts=_contact_evidence(scene,entry),
                source_position_m=np.asarray(d.xpos[entry[1]]).tolist(),
                source_displacement_m=float(np.linalg.norm(d.xpos[entry[1]]-source_pos))))
        scene.raw=scene.backend._get_observations(force_update=True)
        packet=scene.capture_agent()
        if 'initial_occupancy' in packet.get('robot_state',{}):
            raise InitializationError('Stale initial EMPTY fact survived physical initialization')
        images=[]
        for view in PUBLIC_VIEWS:
            rgb=np.asarray(packet['cameras'][view]['rgb'])
            if rgb.shape!=(480,640,3):raise InitializationError('Actual public camera must be 640x480 RGB')
            path=output/(view+'.png');Image.fromarray(rgb).save(path)
            images.append(dict(camera=view,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                width=640,height=480,executed_sequence=int(scene.sequence)))
        receipt['images']=images
        final=_body_qpos(scene)
        receipt['nonrobot_pose_changes_after_physics']={k:float(np.linalg.norm(v[:3]-final[k][:3])) for k,v in before.items()}
        samples=receipt['private_samples'][-STABLE_HOLD_CONTROLS:]
        positions=np.asarray([s['source_position_m'] for s in samples])
        span=float(np.max(np.linalg.norm(positions-positions[-1],axis=1)))
        displacement=float(np.linalg.norm(d.xpos[entry[1]]-source_pos))
        quats=np.asarray([s['source_quaternion_wxyz'] for s in samples])
        rotation_span=float(np.max(2*np.arccos(np.clip(np.abs(quats@quats[-1]),0,1))))
        no_person_contact=not any(s['contacts']['penetrating_robot_person_contacts'] for s in receipt['private_samples'])
        receipt['qualification']=dict(actual_close_controls=MAX_CLOSE_CONTROLS,
            stable_held_controls=STABLE_HOLD_CONTROLS,all_final_samples_assist_held=all(s['held'] for s in samples),
            final_position_span_m=span,final_rotation_span_rad=rotation_span,total_source_displacement_m=displacement,
            no_robot_person_penetrating_contact_during_closure=no_person_contact,
            tabletop_max_vertical_change_m=float(np.max(np.abs(positions[:,2]-source_pos[2]))),
            original_assist_tolerances_unchanged=True, no_object_qpos_writes=True,
            sequence=int(scene.sequence),event_trigger_time_s=getattr(scene,'trigger_time_s',None))
        passed=(all(s['held'] for s in samples) and span <= .003 and rotation_span <= .03 and no_person_contact and displacement<=.015
                and float(np.max(np.abs(positions[:,2]-source_pos[2])))<=.010
                and getattr(scene,'trigger_time_s',None) is None)
        if not passed:raise InitializationError('Physical CLOSE did not leave a stable tabletop initial grasp')
        receipt['status']='PASS'
        # Only public actual records exposed through this explicit method.
        scene.preheld_public_execution_samples=lambda:deepcopy(receipt['public_samples'])
        scene.preheld_initialization_version=VERSION
        scene.preheld_public_initialization_identity=deepcopy(identity)
        return receipt
    except Exception as exc:
        receipt['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['elapsed_s']=time.monotonic()-started
        _write(output/'receipt.json',receipt)


def replay_initial_execution(adapter, receipt):
    """Bind initialization to this exact reset and replay public actual records.

    A B0 adapter has no gate estimator: validate identity/sequence and retain
    public initialization provenance only. An Ours adapter must be fresh.
    """
    if receipt.get('status')!='PASS' or receipt.get('version')!=VERSION:
        raise ValueError('A successful current initial-condition receipt is required')
    scene=adapter.scene
    identity=getattr(scene,'preheld_public_initialization_identity',None)
    if identity is None or identity!=receipt.get('identity'):
        raise ValueError('Initialization receipt is not bound to this exact scene reset')
    if getattr(adapter,'episode',identity['episode'])!=identity['episode']:
        raise ValueError('Initialization receipt episode does not match adapter')
    actual=getattr(scene,'preheld_public_execution_samples',lambda:None)()
    samples=receipt['public_samples']
    if actual!=samples or not samples or scene.sequence!=samples[-1]['executed_sequence']:
        raise ValueError('Initialization receipt public execution sequence mismatch')
    estimator=getattr(adapter,'estimator',None)
    result=(replay_public_samples(estimator,samples) if estimator is not None else
        dict(holding_estimate='not_applicable_B0',last_gripper_command='CLOSE',
             executed_sequence=int(scene.sequence),private_latch_used=False,forced_occupancy=False))
    result.update(initialization_identity=deepcopy(identity),
                  public_sample_count=len(samples),version=VERSION)
    adapter.preheld_initialization_public_provenance=deepcopy(result)
    return result
