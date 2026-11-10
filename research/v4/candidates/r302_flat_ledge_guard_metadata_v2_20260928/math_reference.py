"""Saved-MJB metadata mathematics only; no Scene, MjData or forward."""
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.accelerated import tabletop

H=Path(__file__).resolve().parent;ROOT=H.parents[3]
OLD=ROOT/'research/v4/candidates/r302_source_clearance_ten_seed_prepare_20260928/root_qualification_seed0_9/correct_seed2/startup_seed0/compiled_model.mjb'
NEW=ROOT/'research/v4/candidates/r302_flat_ledge_root_qualifier_20260928/root_actual_seed2_trio/seed2_correct/actual_sequence0.mjb'
PREFIX=bytes(str(ROOT/'envs/sim-smoke/lib/python3.11/site-packages/robosuite/'),'utf8')
REPLACEMENT=bytes(str(ROOT/'envs/libero/lib/python3.11/site-packages/robosuite/'),'utf8')
BASE_XML=ROOT/'envs/libero/lib/python3.11/site-packages/robosuite/models/assets/base.xml'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def pure_qpos0_COM(m):
    """Quaternion FK and reverse mass accumulation, not a simulation call.

    At qpos0 all hinge/slide reference displacements are zero. Free bodies
    use their original qpos0 position/quaternion. Reciprocal multiplication
    matches MuJoCo's scale3 arithmetic; no live pose or data is written.
    """
    p=np.zeros((m.nbody,3));Q=np.tile(np.array([1.,0,0,0]),(m.nbody,1));R=np.tile(np.eye(3),(m.nbody,1,1))
    for i in range(1,m.nbody):
        par=int(m.body_parentid[i]);v=np.empty(3);mujoco.mju_mulMatVec3(v,R[par].reshape(9),m.body_pos[i]);p[i]=p[par]+v
        mujoco.mju_mulQuat(Q[i],Q[par],m.body_quat[i]);mujoco.mju_quat2Mat(R[i].reshape(9),Q[i])
        for j in range(int(m.body_jntadr[i]),int(m.body_jntadr[i]+m.body_jntnum[i])):
            if m.jnt_type[j]==int(mujoco.mjtJoint.mjJNT_FREE):
                if par!=0:raise RuntimeError('Unverified nonworld free-body reference')
                a=int(m.jnt_qposadr[j]);p[i]=m.qpos0[a:a+3];Q[i]=m.qpos0[a+3:a+7];mujoco.mju_quat2Mat(R[i].reshape(9),Q[i])
            elif m.jnt_type[j]not in (int(mujoco.mjtJoint.mjJNT_SLIDE),int(mujoco.mjtJoint.mjJNT_HINGE)):
                raise RuntimeError('Unverified ball reference, fail closed')
    inertial=np.empty_like(p)
    for i in range(m.nbody):
        v=np.empty(3);mujoco.mju_mulMatVec3(v,R[i].reshape(9),m.body_ipos[i]);inertial[i]=p[i]+v
    weighted=inertial*m.body_mass[:,None];mass=m.body_mass.copy()
    for i in range(m.nbody-1,0,-1):
        parent=int(m.body_parentid[i]);weighted[parent]+=weighted[i];mass[parent]+=mass[i]
    return weighted[0]*(1./mass[0]),mass,p,Q,inertial


def path_reference(m):
    raw=m.paths;expected=raw.replace(PREFIX,REPLACEMENT)
    rows=[];fields={}
    for kind in ('mesh','tex','hfield','skin'):
        field=kind+'_pathadr';arr=getattr(m,field).copy();names={'tex':'texture'}.get(kind,kind)
        for i,adr in enumerate(arr):
            if adr<0:continue
            old=raw[int(adr):].split(b'\0',1)[0]
            new=old.replace(PREFIX,REPLACEMENT)
            op=Path(old.decode());npth=Path(new.decode())
            if not op.is_file() or not npth.is_file() or sha(op)!=sha(npth):raise RuntimeError('Asset source path/content mismatch '+str(op))
            arr[i]=len(raw[:int(adr)].replace(PREFIX,REPLACEMENT))
            rows.append(dict(kind=kind,id=i,name=getattr(m,names)(i).name,
                old_path=str(op),new_path=str(npth),old_sha256=sha(op),new_sha256=sha(npth),
                original_adr=int(adr),derived_adr=int(arr[i]),changed_path=old!=new))
        fields[field]=arr
    return expected,fields,rows


def reference(seed):
    from xml.etree import ElementTree as ET
    if ET.parse(BASE_XML).getroot().find('compiler').get('inertiagrouprange')!='0 0':
        raise RuntimeError('Exact production compiler group-range changed')
    m=mujoco.MjModel.from_binary_path(str(OLD));w=m.body('r302_v_wood_cradle').id
    if m.body_parentid[w]!=0 or m.body_jntnum[w]!=0:raise RuntimeError('Wood is not an original fixed body')
    pos,yaw=tabletop.sample_slot('L',int(seed),object_index=0)
    pos[:2]+=np.asarray((0.,.08));pos[2]=tabletop.TABLE_Z;pos[1]+=-.312266002685539-(-.16)
    q=Rotation.from_euler('z',math.pi)*Rotation.from_quat([0.,0.,math.sin(yaw/2),math.cos(yaw/2)])
    x,y,z,wq=q.as_quat();quat=np.asarray([float(f'{v:.17g}')for v in (wq,x,y,z)]);mujoco.mju_normalize4(quat)
    m.body_pos[w]=pos;m.body_quat[w]=quat
    # Group2 is outside the actual compiler's group0 inertial range. This
    # fixed-body default representation is exact, not a mass/force repair.
    m.body_mass[w]=0.;m.body_inertia[w]=0.;m.body_ipos[w]=pos;m.body_iquat[w]=quat
    com,subtree,_,_,_=pure_qpos0_COM(m);m.body_subtreemass[:]=subtree
    for prefix,count in [('cam',m.ncam),('light',m.nlight)]:
        for i in range(count):
            if getattr(m,prefix+'_bodyid')[i]==0:
                getattr(m,prefix+'_poscom0')[i]=getattr(m,prefix+'_pos0')[i]-com
    # The original runtime microwave fixture uses its own sample_slot(R),
    # translation, yaw and mat2Quat normalization. Only that named row moves.
    cfg=json.loads((ROOT/'research/v4/candidates/r3_02_cc0_soup_config_20260926/config_v1.json').read_text())
    trans=json.loads((ROOT/'research/v4/candidates/r3_02_oven_only_prefix_20260928/layout.json').read_text())['translation']
    p,angle=tabletop.sample_slot(cfg['microwave']['slot'],int(seed),object_index=1)
    p[:2]+=np.asarray(cfg['microwave']['xy_offset']);p[0]+=cfg['layout']['oven_dx_m'];p+=np.asarray(trans)
    ww,zz=math.cos(angle/2),math.sin(angle/2);c,s=math.cos(-math.pi/4),math.sin(-math.pi/4)
    oq=np.asarray([c*ww-s*zz,0.,0.,c*zz+s*ww]);mat=np.empty(9);outq=np.empty(4)
    mujoco.mju_quat2Mat(mat,oq);mujoco.mju_mat2Quat(outq,mat)
    e=m.equality('r3_microwave_fixture').id;m.eq_data[e,3:6]=p;m.eq_data[e,6:10]=outq
    paths,addresses,rows=path_reference(m)
    for field,values in addresses.items():getattr(m,field)[:]=values
    return m,dict(seed=int(seed),ACTUAL_MJB=str(OLD),compiler_group_range=[0,0],
        fixed_group2_wood_mass_kg=0.,fixed_group2_wood_ipos=pos.tolist(),fixed_group2_wood_iquat=quat.tolist(),
        qpos0_world_COM=com.tolist(),mass_total_kg=float(subtree[0]),path_rows=rows,
        changed_paths=sum(r['changed_path']for r in rows),path_byte_delta=len(paths)-m.npaths,
        no_Scene_MjData_compile_forward_or_controls=True,no_live_model_or_qpos_writes=True,
        original_inactive_assist_row_untouched=True),paths
