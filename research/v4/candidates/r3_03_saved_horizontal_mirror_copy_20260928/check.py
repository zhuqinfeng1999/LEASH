"""One own R303 saved-frame continuous copied route; no Scene or controls."""
import json,hashlib,inspect,textwrap,types
from pathlib import Path
from types import SimpleNamespace
import numpy as np,mujoco
from research.v4.candidates.r3_04_source_axis_low_entry_copy_20260928 import check as original
from research.v4.candidates.r3_04_source_axis_mirrored_low_copy_20260928 import check as mirrored
from research.v4.candidates.r3_04_mirrored_initial_saved_copy_20260928.check import no_restart_ik
from research.v4.candidates.r3_04_source_axis_grasp_proposal_20260928 import contacts as padhelper
from research.v4.accelerated.grasp_assist import GraspAssist
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
SAVED=H.parent/'r3_03_front_initial_constructor_20260928'
POINT=(0.,0.,.059)

def frame(seed):
    if seed!=0:raise ValueError('Own exact saved seed0 only')
    receipt=json.loads((SAVED/'receipt.json').read_text())
    if not receipt['passed']:raise ValueError('Saved actual own constructor failed')
    for name,sha in receipt['saved_files'].items():
        if hashlib.sha256((SAVED/name).read_bytes()).hexdigest()!=sha:raise ValueError('Actual own saved input drift '+name)
    m=mujoco.MjModel.from_binary_path(str(SAVED/'compiled_model.mjb'));d=mujoco.MjData(m)
    saved=np.load(SAVED/'constructor_data.npz',allow_pickle=False)
    mujoco.mj_setState(m,d,saved['integration_state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    assert np.array_equal(d.qpos,saved['qpos'])
    arm=[m.joint('robot0_joint'+str(i)).id for i in range(1,8)]
    assert np.array_equal(d.qpos[m.jnt_qposadr[arm]],receipt['initial_robot_parameters'])
    kwargs={'hand_body':'gripper0_right_eef','center_site':'gripper0_right_grip_site','left_pad_site':'v42_left_pad','right_pad_site':'v42_right_pad','slot_metadata':{'L':{'grasp_zones':[{'object_body':'steel_thermos_1_main','zone_site':'v42_steel_thermos_1_grasp_zone','weld_name':'v42_steel_thermos_1_assist_weld','tolerance_xyz':(.015,.015,.020)}]}}}
    assist=GraspAssist(m,d,**kwargs)
    evidence={'source':'actual own R303 saved root constructor, no new Scene','initial_joint7':float(d.qpos[m.jnt_qposadr[arm[-1]]]),'initial_joint_margins':mirrored.joint_margins(m,d,arm),'initial_EEF':d.site_xpos[assist._center].tolist(),'initial_R':d.site_xmat[assist._center].reshape(3,3).tolist(),'source_world':d.xpos[m.body('steel_thermos_1_main').id].tolist(),'grasp_site_local':m.site_pos[m.site('v42_steel_thermos_1_grasp_zone').id].tolist(),'source_mass':float(m.body_mass[m.body('steel_thermos_1_main').id]),'initial_saved_qpos_exact':True,'Scene_controls':0}
    (H/'initial_frame_receipt.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return SimpleNamespace(backend=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=m),data=SimpleNamespace(_data=d)),objects_dict={'steel_thermos_1':SimpleNamespace(root_body='steel_thermos_1_main',joints=['steel_thermos_1_joint0']),'microwave_1':SimpleNamespace(root_body='microwave_1_main')},grasp_assist_kwargs=kwargs),grasp_assist=assist,item_name='steel_thermos_1',oven_name='microwave_1',close=lambda:None)

def exact_pad_binding(records):
    padhelper.bind_actual_compiled_pads(records)
    ns=dict(padhelper.classify.__globals__);ns['MATERIAL']='steel_thermos_1_thermos_steel_body_collision'
    return types.FunctionType(padhelper.classify.__code__,ns,padhelper.classify.__name__)

def contacts(s):
    src=inspect.getsource(mirrored.full_contacts).replace("'foil_sandwich_1_'","'steel_thermos_1_'")
    ns=dict(mirrored.full_contacts.__globals__);exec(compile(src,str(H/'check.py')+'::own-full-contact-identity','exec'),ns)
    return ns['full_contacts'](s)

def support_bounds(m,d,solids,origin,A,Rentry,axes):
    """Exact support of actual cylinders/boxes and all actual mesh vertices."""
    lo=np.full(3,np.inf);hi=np.full(3,-np.inf)
    for g in solids:
        typ=int(m.geom_type[g]);p=A.T@(d.geom_xpos[g]-origin);Q=A.T@d.geom_xmat[g].reshape(3,3)
        for i in range(3):
            direction=Rentry.T@axes[:,i];centre=float(direction@p)
            if typ==5:
                c=Q.T@direction;r,h=m.geom_size[g,:2];extent=float(r*np.hypot(c[0],c[1])+h*abs(c[2]));bounds=(centre-extent,centre+extent)
            elif typ==6:
                extent=float(np.abs(Q.T@direction)@m.geom_size[g]);bounds=(centre-extent,centre+extent)
            elif typ==7:
                mid=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);v=p+(Q@m.mesh_vert[start:start+n].T).T;projections=v@direction;bounds=(float(projections.min()),float(projections.max()))
            else:raise ValueError('Unknown actual support shape; no guessed cylinder/AABB '+str(typ))
            lo[i]=min(lo[i],bounds[0]);hi[i]=max(hi[i],bounds[1])
    return np.stack([lo,hi])

def selected_source():
    src=mirrored.selected_source()
    replacements=[
      ("scene=HorizontalSideFoilScene(0);report['Scene_CPU_constructors']=1;m=scene.backend.sim.model._model;report['exact_constructor_snapshot']=save_exact_constructor(scene,m);d=mujoco.MjData(m);", "scene=frame(0);report['Scene_CPU_constructors']=0;report['saved_own_model_loads']=1;m=scene.backend.sim.model._model;d=mujoco.MjData(m);"),
      ("'instance':'R3-04'", "'instance':'R3-03'"),
      ("v42_foil_sandwich_1_grasp_zone", "v42_steel_thermos_1_grasp_zone"),
      ("np.array(SIDE_POINT)","np.array(POINT)"),
      ("classify=bind_actual_compiled_pads(records)","classify=exact_pad_binding(records)"),
      ("contact=full_contacts;ik,src=instrumented_ik();","contact=contacts;ik,src=no_restart_ik();"),
      ("sourceEEF=sourceR@R0@np.diag([-1.,-1.,1.]);retreat=.001821250654;grip=d.site_xpos[zone].copy()-sourceR[:,0]*retreat;", "sourceEEF=R0@np.diag([-1.,-1.,1.]);retreat=.009462150692939753;grip=d.site_xpos[zone].copy()-sourceEEF[:,2]*retreat;"),
      ("pre=grip-sourceR[:,0]*.06", "pre=grip-sourceEEF[:,2]*.06"),
      ("solids=[g for g in range(m.ngeom)if(m.geom(g).name or '').startswith(('foil_sandwich_1_','gripper0_'))and(m.geom_contype[g]or m.geom_conaffinity[g])];V=np.concatenate([vertices(g)for g in solids]);localhand=(V-origin)@A", "solids=[g for g in range(m.ngeom)if (m.geom(g).name or '').startswith('steel_thermos_1_')or((m.geom(g).name or '').startswith('gripper0_')and(m.geom_contype[g]or m.geom_conaffinity[g]))]"),
      ("entryoffset=(Rentry@localhand.T).T", "entrybounds=support_bounds(m,d,solids,origin,A,Rentry,np.eye(3))"),
      ("entryoffset[:,2].min()", "entrybounds[0,2]"),
      ("entryoffset[:,2].max()", "entrybounds[1,2]"),
      ("full=(Rentry@localhand.T).T@oR;", "fullbounds=support_bounds(m,d,solids,origin,A,Rentry,oR);"),
      ("full[:,0].min()", "fullbounds[0,0]"),
      ("full[:,0].max()", "fullbounds[1,0]"),
      ("all actual compiled collision mesh vertices and exact box corners; no hidden item coordinates online", "own complete thermos visual/collision exact-cylinder support and box corners plus all actual hand collision mesh vertices; no hidden coordinates online")
    ]
    for old,new in replacements:
        if src.count(old)!=1:raise ValueError('Own exact source seam drift '+old)
        src=src.replace(old,new)
    return src

def main():
    src=selected_source()
    if (H/'executed_source.py.txt').read_text()!=src:raise ValueError('Prepared own source changed')
    ns=dict(original.main.__globals__);ns.update(H=H,ROOT=ROOT,POINT=POINT,frame=frame,exact_pad_binding=exact_pad_binding,contacts=contacts,no_restart_ik=no_restart_ik,joint_margins=mirrored.joint_margins,support_bounds=support_bounds)
    exec(compile(src,str(H/'check.py')+'::one-own-R303-saved-horizontal-route','exec'),ns);ns['main']()
if __name__=='__main__':main()
