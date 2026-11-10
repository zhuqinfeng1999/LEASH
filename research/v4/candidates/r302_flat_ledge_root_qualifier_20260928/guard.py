"""Root-only named invariant admission against the ACTUAL stopped seed2 MJB."""
import json
import re
from pathlib import Path

import mujoco
import numpy as np

H=Path(__file__).resolve().parent
ROOT=H.parents[3]
PARENT=ROOT/'research/v4/candidates/r302_flat_ledge_source_20260928'
ACTUAL=ROOT/'research/v4/candidates/r302_source_clearance_ten_seed_prepare_20260928/root_qualification_seed0_9/correct_seed2/startup_seed0/compiled_model.mjb'
HEADER=ROOT/'envs/libero/lib/python3.11/site-packages/mujoco/include/mujoco/mjxmacro.h'
WOOD=('r302_v_base','r302_v_rail_-1','r302_v_rail_1','r302_ledge_end_-1','r302_ledge_end_1')


def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def boxes():
    p=json.loads((PARENT/'metadata_v2/detached_compile.json').read_text())['layout']
    z=p['floor_top_local_m'];a,b=p['inner_half_footprint_m'];ao,bo=p['outer_half_footprint_m']
    return {WOOD[0]:([ao,bo,z/2],[0,0,z/2]),
        WOOD[1]:([a,.0015,.003],[0,-(b+.0015),z+.003]),
        WOOD[2]:([a,.0015,.003],[0,b+.0015,z+.003]),
        WOOD[3]:([.0015,b,.003],[-(a+.0015),0,z+.003]),
        WOOD[4]:([.0015,b,.003],[a+.0015,0,z+.003])}


def support_inertia():
    rows=[(np.asarray(s),np.asarray(p))for s,p in boxes().values()]
    masses=np.asarray([1000*8*np.prod(s)for s,p in rows]);mass=float(masses.sum())
    com=sum(w*p for w,(s,p)in zip(masses,rows))/mass
    I=np.zeros((3,3))
    for w,(s,p)in zip(masses,rows):
        r=p-com;I+=np.diag(w/3*np.asarray([s[1]**2+s[2]**2,s[0]**2+s[2]**2,s[0]**2+s[1]**2]))+w*(np.dot(r,r)*np.eye(3)-np.outer(r,r))
    return mass,com,I


def reference(seed):
    # Both sources are real MJBs; no XML serialization/recompile is admitted.
    from research.v4.candidates.r302_seeded_model_derived_guard_v2_20260928 import guard as seeded
    m=mujoco.MjModel.from_binary_path(str(ACTUAL));other,receipt=seeded.expected_model(seed)
    w=m.body('r302_v_wood_cradle').id
    m.body_pos[w]=other.body_pos[w];m.body_quat[w]=other.body_quat[w]
    e=m.equality('r3_microwave_fixture').id;m.eq_data[e]=other.eq_data[e]
    mass,com,I=support_inertia()
    m.body_mass[w]=mass;m.body_ipos[w]=com
    # Exact compiler's reverse tree accumulation, verified on saved actual
    # seed2: all old body_subtreemass bytes match this order. Avoid rounding
    # of subtract-old/add-new on the heavily weighted world subtree.
    subtree=m.body_mass.copy()
    for j in range(m.nbody-1,0,-1):subtree[int(m.body_parentid[j])]+=subtree[j]
    m.body_subtreemass[:]=subtree
    d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    for prefix,count in [('cam',m.ncam),('light',m.nlight)]:
        for i in range(count):
            if getattr(m,prefix+'_bodyid')[i]==0:
                getattr(m,prefix+'_poscom0')[i]=getattr(m,prefix+'_pos0')[i]+d.xpos[0]-d.subtree_com[0]
    return m,dict(seed=int(seed),ACTUAL_MJB=str(ACTUAL),source_seed=2,
        declared_wood_mass_kg=mass,declared_wood_COM_local_m=com.tolist(),
        declared_wood_inertia_tensor_kg_m2=I.tolist(),seed_derivation=receipt,
        no_XML_recompile=True,no_live_model_or_qpos_writes=True,no_forward_or_controls=True)


def compare(actual,seed):
    old,receipt=reference(seed);w=old.body('r302_v_wood_cradle').id
    errors=[];checked=[];allowed=[]
    def check(label,a,b):
        if not same(a,b):errors.append(label)
        else:checked.append(label)
    # Every old named object's ID and name must survive. Only two new end boxes
    # and deletion of the two exact obsolete wedge pairs are permitted.
    kinds=('body','geom','joint','site','camera','light','actuator','sensor','equality','mesh','material','texture','tendon','exclude','numeric','text','tuple','key','skin','hfield','flex','plugin')
    for kind in kinds:
        try:count={'joint':old.njnt,'camera':old.ncam,'material':old.nmat,'texture':old.ntex,'equality':old.neq}.get(kind,getattr(old,'n'+kind,0))
        except AttributeError:count=0
        for i in range(count):
            before=getattr(old,kind)(i);after=getattr(actual,kind)(i)
            if before.name!=after.name:errors.append(kind+'.name/id.'+str(i))
            if before.name and getattr(actual,kind)(before.name).id!=i:errors.append(kind+'.hash_lookup/id.'+str(i))
    if actual.ngeom!=old.ngeom+2 or actual.npair!=0 or old.npair!=2:errors.append('exact geom+2/pair-2 dimensions')
    if [old.pair(i).name for i in range(old.npair)]!=['r302_source_pair_-1','r302_source_pair_1']:errors.append('old pair names')
    if [actual.geom(i).name for i in range(old.ngeom,actual.ngeom)]!=list(WOOD[3:]):errors.append('exact appended ends')
    # Source support is one fixed body. It must not become a free/welded fixture.
    if actual.body_jntnum[w]!=0 or actual.body_parentid[w]!=0:errors.append('wood fixed/table-supported body')
    mass,com,I=support_inertia();q=actual.body_iquat[w];R=np.empty(9);mujoco.mju_quat2Mat(R,q);R=R.reshape(3,3)
    if not (abs(actual.body_mass[w]-mass)<=1e-14 and np.allclose(actual.body_ipos[w],com,atol=1e-14,rtol=0)
        and np.allclose(R@np.diag(actual.body_inertia[w])@R.T,I,atol=1e-14,rtol=0)):
        errors.append('exact five-box density1000 wood inertial derivation')
    # This numerical 1e-14 certificate concerns compilation arithmetic only;
    # it is never a contact, rest, grasp or benchmark tolerance.
    special={'body_inertia','body_iquat','body_geomnum','body_bvhnum','body_bvhadr','body_geomadr',
             'mesh_bvhadr','cam_poscom0','light_poscom0','_sizes','names_map'}
    geometry_changes={'geom_size','geom_pos','geom_quat','geom_aabb','geom_rbound','geom_group','geom_sameframe'}
    for field in dir(old):
        a,b=getattr(old,field,None),getattr(actual,field,None)
        if not isinstance(a,np.ndarray):continue
        if not isinstance(b,np.ndarray):errors.append(field+'.missing');continue
        if field.startswith('pair_'):
            if b.shape[0]!=0:errors.append(field+'.removed_exact_pairs_not_empty')
            allowed.append(field+': exact obsolete2 ->0');continue
        if field.startswith('name_'):
            allowed.append(field+': decoded ID/name identities checked');continue
        if field.startswith('bvh_') or field in special:continue
        if field.startswith('geom_'):
            for i in range(old.ngeom):
                name=old.geom(i).name
                if name in WOOD[:3] and field in geometry_changes:continue
                check(field+':'+str(name),a[i],b[i])
            # New ends inherit original box defaults, with exact source RGBA.
            for name in WOOD[3:]:
                i=actual.geom(name).id;base=old.geom(WOOD[0]).id
                if field in geometry_changes:continue
                expected=a[base]
                if field=='geom_rgba':expected=np.asarray([.52,.28,.11,1],dtype=a.dtype)
                check(field+':'+name,expected,b[i])
        elif field.startswith('body_') and field in {'body_mass','body_ipos'}:
            mask=np.arange(old.nbody)!=w;check(field+'.nonwood',a[mask],b[mask])
        else:check(field,a,b)
    for name,(size,pos)in boxes().items():
        i=actual.geom(name).id
        check('wood.size:'+name,np.asarray(size),actual.geom_size[i]);check('wood.pos:'+name,np.asarray(pos),actual.geom_pos[i])
        check('wood.quat:'+name,np.asarray([1.,0,0,0]),actual.geom_quat[i])
        check('wood.aabb:'+name,np.r_[np.zeros(3),size],actual.geom_aabb[i])
        check('wood.rbound:'+name,np.asarray(np.linalg.norm(size)),np.asarray(actual.geom_rbound[i]))
        if actual.geom_group[i]!=2 or actual.geom_type[i]!=int(mujoco.mjtGeom.mjGEOM_BOX) or actual.geom_bodyid[i]!=w:
            errors.append('visible collision box identity:'+name)
        if actual.geom_sameframe[i]!=int(mujoco.mjtSameFrame.mjSAMEFRAME_BODYROT):errors.append('wood compiler sameframe:'+name)
    for field in ('body_inertia','body_iquat'):
        mask=np.arange(old.nbody)!=w;check(field+'.nonwood',getattr(old,field)[mask],getattr(actual,field)[mask])
    for field in ('body_geomadr','body_geomnum','body_bvhnum','body_bvhadr'):
        mask=np.arange(old.nbody)!=w;check(field+'.nonwood',getattr(old,field)[mask],getattr(actual,field)[mask])
    if actual.body_geomadr[w]!=old.body_geomadr[w] or actual.body_geomnum[w]!=5 or actual.body_bvhnum[w]!=9 or actual.body_bvhadr[w]!=old.body_bvhadr[w]:errors.append('wood exact3->5 geoms/5->9 BVH allocation')
    # Unchanged body and mesh BVH subtrees: compare relative topology, leaves,
    # depth and boxes, thereby admitting only insertion of4 wood BVH nodes.
    for category,count in [('body',old.nbody),('mesh',old.nmesh)]:
        for i in range(count):
            if category=='body' and i==w:continue
            adr=int(getattr(old,category+'_bvhadr')[i]);num=int(getattr(old,category+'_bvhnum')[i]);newadr=int(getattr(actual,category+'_bvhadr')[i])
            check(category+'.bvhnum:'+str(i),np.asarray(num),np.asarray(int(getattr(actual,category+'_bvhnum')[i])))
            if num==0:continue
            for field in ('bvh_aabb','bvh_depth','bvh_nodeid','bvh_child'):
                aa=getattr(old,field)[adr:adr+num].copy();bb=getattr(actual,field)[newadr:newadr+num].copy()
                check(category+'.'+field+':'+str(i),aa,bb)
    # MuJoCo stores BVH children as subtree-relative indices. Wood boxes are
    # in the body's inertial frame; verify every leaf and every union box.
    adr=int(actual.body_bvhadr[w]);num=int(actual.body_bvhnum[w]);leaves=[]
    for k in range(num):
        node=adr+k;gid=int(actual.bvh_nodeid[node]);children=actual.bvh_child[node]
        if gid>=0:
            name=actual.geom(gid).name;leaves.append(name)
            if name not in boxes():errors.append('wood BVH unexpected leaf');continue
            size,pos=boxes()[name];center=R.T@(np.asarray(pos)-actual.body_ipos[w]);half=np.abs(R.T)@np.asarray(size)
            expected=np.r_[center,half]
        else:
            if any(x<0 or x>=num for x in children):errors.append('wood BVH invalid children');continue
            rows=actual.bvh_aabb[adr+children];lo=np.min(rows[:,:3]-rows[:,3:],axis=0);hi=np.max(rows[:,:3]+rows[:,3:],axis=0);expected=np.r_[(hi+lo)/2,(hi-lo)/2]
        if not np.allclose(actual.bvh_aabb[node],expected,atol=1e-12,rtol=0):errors.append('wood BVH derived box:'+str(k))
    if sorted(leaves)!=sorted(WOOD):errors.append('wood BVH five exact leaves')
    check('cam_poscom0 exact mass/seed-derived',old.cam_poscom0,actual.cam_poscom0)
    check('light_poscom0 exact mass/seed-derived',old.light_poscom0,actual.light_poscom0)
    # Every SDK dimension, including _sizes, is explicit; names/buffer are
    # allocator metadata, not exemptions of entity/dynamics arrays.
    text=HEADER.read_text().split('#define MJMODEL_INTS',1)[1].split('/* nbuffer',1)[0]
    dims=re.findall(r'(?:X|XMJV)\s*\(\s*(\w+)\s*\)',text)
    size_changes=[]
    name_delta=sum(len(n)+1 for n in WOOD[3:])-sum(len(n)+1 for n in ('r302_source_pair_-1','r302_source_pair_1'))
    # The explicit same two named records replaced by two boxes keep the
    # object hash allocation identical; saved source compile shows equal
    # buffer bytes. Only exactly -4 name bytes and +4 BVH nodes are admitted.
    allowed_dims={'ngeom':old.ngeom+2,'npair':0,'nbvh':old.nbvh+4,'nbvhstatic':old.nbvhstatic+4,
                  'nnames':old.nnames+name_delta,'nnames_map':old.nnames_map,'nbuffer':old.nbuffer}
    for n in dims:
        x,y=int(getattr(old,n)),int(getattr(actual,n))
        if x!=y:size_changes.append(dict(field=n,before=x,after=y))
        if n in allowed_dims:
            if y!=allowed_dims[n]:errors.append('dimension:'+n)
        elif x!=y:errors.append('dimension:'+n)
    if len(old._sizes)!=len(dims) or len(actual._sizes)!=len(dims):errors.append('SDK _sizes/header identity')
    else:
        for i,n in enumerate(dims):
            if int(old._sizes[i])!=int(getattr(old,n)) or int(actual._sizes[i])!=int(getattr(actual,n)):errors.append('_sizes field:'+n)
    for n in dir(old.opt):
        x,y=getattr(old.opt,n,None),getattr(actual.opt,n,None)
        if isinstance(x,(int,float,np.ndarray)):check('opt.'+n,x,y)
    return dict(passed=not errors,errors=errors,checked_fields=checked,explicit_pair_allocation=allowed,
        allocator_changes=size_changes,reference=receipt,actual_wood=dict(mass=float(actual.body_mass[w]),ipos=actual.body_ipos[w].tolist(),inertia=actual.body_inertia[w].tolist(),iquat=actual.body_iquat[w].tolist()),
        unchanged_can_robot_microwave_dynamic_fields_byteexact=not errors,
        no_live_pose_writes=True,guard_before80_required=True,support_force_and_rest_not_proven_by_model=True)
