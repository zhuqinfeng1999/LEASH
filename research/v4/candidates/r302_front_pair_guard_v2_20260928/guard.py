"""Exact original ID/name, physical arrays and explained pair buffer allocation."""
import json
import re
from pathlib import Path
import mujoco
import numpy as np

H=Path(__file__).resolve().parent
ROOT=H.parents[3]
HEADER=ROOT/'envs/sim-smoke/lib/python3.11/site-packages/mujoco/include/mujoco/mjmodel.h'
# All installed model name-pointer categories, not selected task nouns.
CATEGORIES=(('body','nbody','BODY'),('jnt','njnt','JOINT'),('geom','ngeom','GEOM'),('site','nsite','SITE'),
 ('cam','ncam','CAMERA'),('light','nlight','LIGHT'),('flex','nflex','FLEX'),('mesh','nmesh','MESH'),
 ('skin','nskin','SKIN'),('hfield','nhfield','HFIELD'),('tex','ntex','TEXTURE'),('mat','nmat','MATERIAL'),
 ('pair','npair','PAIR'),('exclude','nexclude','EXCLUDE'),('eq','neq','EQUALITY'),('tendon','ntendon','TENDON'),
 ('actuator','nu','ACTUATOR'),('sensor','nsensor','SENSOR'),('numeric','nnumeric','NUMERIC'),('text','ntext','TEXT'),
 ('tuple','ntuple','TUPLE'),('key','nkey','KEY'),('plugin','nplugin','PLUGIN'))
def decode(m,adr):return m.names[int(adr):].split(b'\0',1)[0].decode('utf-8')
def arrays(m):return {k:getattr(m,k) for k in dir(m) if isinstance(getattr(m,k,None),np.ndarray)}
def buffer_bytes(m):
    # Observed official pointer order aligns every array/name block to64;
    # last paths block is not rounded. Validate independently for both models.
    data=arrays(m)
    alignment=all((a.ctypes.data-m.qpos0.ctypes.data)%64==0 for n,a in data.items() if n!='_sizes' and a.size)
    size=sum(((a.nbytes+63)//64)*64 for n,a in data.items() if n!='_sizes')+((m.nnames+63)//64)*64+m.npaths
    return {'array_and_names64_aligned':alignment,'calculated_bytes':int(size),'declared_bytes':int(m.nbuffer),'passed':bool(alignment and size==m.nbuffer)}
def compare(previous,current,params):
    failures=[];aa,bb=arrays(previous),arrays(current)
    changed=[n for n in aa if n not in bb or aa[n].shape!=bb[n].shape or aa[n].dtype!=bb[n].dtype or aa[n].tobytes()!=bb[n].tobytes()]
    name_fields={f'name_{k}adr' for k,_,_ in CATEGORIES}
    names=[]
    for prefix,dim,enum in CATEGORIES:
        typ=getattr(mujoco.mjtObj,'mjOBJ_'+enum)
        if prefix!='pair' and getattr(previous,dim)!=getattr(current,dim):failures.append('object dimension '+dim)
        arr_old=getattr(previous,'name_'+prefix+'adr');arr_new=getattr(current,'name_'+prefix+'adr')
        for i in range(getattr(previous,dim)):
            x,y=decode(previous,arr_old[i]),decode(current,arr_new[i])
            oldid=mujoco.mj_name2id(previous,typ,x) if x else -1
            newid=mujoco.mj_name2id(current,typ,y) if y else -1
            ok=x==y and oldid==newid and (not x or newid==i)
            names.append({'category':prefix,'id':i,'old_name':x,'new_name':y,'old_address':int(arr_old[i]),'new_address':int(arr_new[i]),'exact_identity':ok})
            if not ok:failures.append('named identity '+prefix+':'+str(i))
    pair_names=['r302_source_pair_'+r['rail'].rsplit('_',1)[-1] for r in params['rows']]
    stripped=current.names
    for n in pair_names:
        token=n.encode()+b'\0'
        if stripped.count(token)!=1:failures.append('new name occurrence '+n)
        stripped=stripped.replace(token,b'')
    if stripped!=previous.names:failures.append('other name buffer bytes changed')
    if current.nnames!=previous.nnames+sum(len(n.encode())+1 for n in pair_names):failures.append('names exact byte length')
    if current.npair!=2 or previous.npair!=0:failures.append('only two new pairs')
    pair_details=[]
    for i in range(current.npair):
        n=current.pair(i).name;g=[current.geom(int(v)).name for v in (current.pair_geom1[i],current.pair_geom2[i])]
        row=next((r for r in params['rows'] if set(g)=={r['can'],r['rail']}),None)
        if row is None or n not in pair_names:failures.append('unexpected pair identity');continue
        checks={'condim':int(current.pair_dim[i])==row['dim'],'solref':np.array_equal(current.pair_solref[i],row['new_solref']),
            'solreffriction':np.array_equal(current.pair_solreffriction[i],row['old_solreffriction']),
            'solimp':np.array_equal(current.pair_solimp[i],row['solimp']),'friction':np.array_equal(current.pair_friction[i],row['friction']),
            'margin':float(current.pair_margin[i])==row['margin'],'gap':float(current.pair_gap[i])==row['gap'],
            'signature':int(current.pair_signature[i])==((int(current.geom_bodyid[current.pair_geom1[i]])<<16)+int(current.geom_bodyid[current.pair_geom2[i]])),
            'name_lookup':mujoco.mj_name2id(current,mujoco.mjtObj.mjOBJ_PAIR,n)==i}
        if not all(checks.values()):failures.append('exact pair field '+n)
        pair_details.append({'name':n,'id':i,'geoms':g,'checks':checks})
    text=HEADER.read_text().split('struct mjModel_ {',1)[1].split('// ------------------------------- options',1)[0]
    size_names=re.findall(r'(?:int|size_t)\s+(\w+)\s*;',text)
    mapped_old=np.asarray([getattr(previous,n) for n in size_names]);mapped_new=np.asarray([getattr(current,n) for n in size_names])
    if not np.array_equal(mapped_old,previous._sizes) or not np.array_equal(mapped_new,current._sizes):failures.append('official sizes field mapping')
    allowed_sizes={'npair','nnames','nnames_map','nbuffer'}
    sizes=[{'index':i,'field':n,'old':int(mapped_old[i]),'new':int(mapped_new[i])} for i,n in enumerate(size_names) if mapped_old[i]!=mapped_new[i]]
    if any(s['field'] not in allowed_sizes for s in sizes):failures.append('unexplained dimension/allocator change')
    old_count=sum(getattr(previous,d) for _,d,_ in CATEGORIES);new_count=sum(getattr(current,d) for _,d,_ in CATEGORIES)
    if previous.nnames_map!=2*old_count or current.nnames_map!=2*new_count or current.nnames_map-previous.nnames_map!=4:failures.append('names map exact two-pair allocation')
    buffers={'old':buffer_bytes(previous),'new':buffer_bytes(current)}
    if not all(v['passed'] for v in buffers.values()):failures.append('exact buffer byte accounting')
    explained={n for n in changed if n.startswith('pair_') or n in name_fields or n in ('_sizes','names_map')}
    unexplained=[n for n in changed if n not in explained]
    if unexplained:failures.append('nonpair physical-array difference '+repr(unexplained))
    if set(aa)!=set(bb):failures.append('unexpected model array category')
    if current.opt.timestep!=params['dt_s']:failures.append('physics dt changed')
    return {'passed':not failures,'failures':failures,'same_model_array_count':len(aa)-len(changed),
        'changed_model_arrays':changed,'unexplained_physical_arrays':unexplained,
        'exact_old_entity_ID_name_checks':names,'compiled_pairs':pair_details,
        'all85_sizes_exact_or_explained':sizes,'exact_buffer_accounting':buffers,
        'names_buffer_old_bytes_after_removing_exact_two_pair_names':stripped==previous.names,
        'names_map_slots_old_new':[previous.nnames_map,current.nnames_map],
        'no_state_restore_or_integration_bitclaim':True}
