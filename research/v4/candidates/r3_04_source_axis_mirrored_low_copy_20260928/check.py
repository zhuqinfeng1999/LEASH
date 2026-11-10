"""One source-axis bilateral mirror, copied geometry only; no actual controls."""
import hashlib, inspect, json, textwrap
from pathlib import Path
import numpy as np, mujoco
from research.v4.candidates.r3_04_source_axis_low_entry_copy_20260928 import check as original
H=Path(__file__).resolve().parent
ROOT=H.parents[3]

def joint_margins(m,d,arm):
    q=d.qpos[m.jnt_qposadr[np.asarray(arm)]].copy()
    ranges=m.jnt_range[np.asarray(arm)]
    lo=q-ranges[:,0]; hi=ranges[:,1]-q
    return {'names':[m.joint(j).name for j in arm], 'qpos_rad':q.tolist(),
            'range_rad':ranges.tolist(), 'lower_margin_rad':lo.tolist(),
            'upper_margin_rad':hi.tolist(), 'minimum_margin_rad':float(min(lo.min(),hi.min())),
            'within_20mrad_indices':[int(i+1) for i in np.flatnonzero(np.minimum(lo,hi)<=.020)],
            'copied_solver_clip_5mrad_is_not_actual_OSC_assurance':True}

def full_contacts(s):
    """All robot/self/environment and foil/environment contacts; exact pads later."""
    m=s.backend.sim.model._model; d=s.backend.sim.data._data; out=[]
    for i,c in enumerate(d.contact[:d.ncon]):
        names=[m.geom(int(g)).name or '' for g in (c.geom1,c.geom2)]
        if not any(n.startswith(('robot0_','gripper0_','foil_sandwich_1_')) for n in names): continue
        f=np.zeros(6); mujoco.mj_contactForce(m,d,i,f)
        out.append({'geoms':names,'dist_m':float(c.dist),'world_position':c.pos.tolist(),
                    'world_contact_frame':c.frame.tolist(),'contact_local_force':f.tolist(),
                    'scope':'all robot/self/environment and foil/environment actual contacts'})
    return out

def save_exact_constructor(scene,m):
    d=scene.backend.sim.data._data
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    before=np.empty(mujoco.mj_stateSize(m,spec)); mujoco.mj_getState(m,d,before,spec)
    arrays={}
    for k in dir(m):
        if k.startswith('_'):continue
        v=getattr(m,k)
        if isinstance(v,np.ndarray):arrays[k]=v.copy()
    np.savez_compressed(H/'compiled_model_arrays.npz',**arrays)
    np.savez_compressed(H/'constructor_data.npz',integration_state=before,
                        qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),eq_active=d.eq_active.copy())
    mujoco.mj_saveModel(m,str(H/'compiled_model.mjb'))
    mujoco.mj_saveLastXML(str(H/'compiled_model.xml'),m)
    after=np.empty_like(before);mujoco.mj_getState(m,d,after,spec)
    unchanged=before.tobytes()==after.tobytes()
    if not unchanged:raise ValueError('Saving original constructor model changed integration state')
    receipt={'compiled_array_count':len(arrays),'integration_byte_exact_after_save':unchanged,
             'sequence':scene.sequence,'sim_time_s':float(d.time),'actual_controls_added':0,
             'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [H/'compiled_model_arrays.npz',H/'constructor_data.npz',H/'compiled_model.mjb',H/'compiled_model.xml']}}
    (H/'constructor_snapshot.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

def selected_source():
    src=textwrap.dedent(inspect.getsource(original.main))
    edits=[
      ('sourceEEF=sourceR@R0;', 'sourceEEF=sourceR@R0@np.diag([-1.,-1.,1.]);'),
      ('Rentry=np.column_stack((-oR[:,0],oR[:,2],oR[:,1]));',
       'Rentry=np.column_stack((oR[:,0],-oR[:,2],oR[:,1]));'),
      ('m=scene.backend.sim.model._model;d=mujoco.MjData(m);',
       "m=scene.backend.sim.model._model;report['exact_constructor_snapshot']=save_exact_constructor(scene,m);d=mujoco.MjData(m);"),
      ('contact=selected_contacts();', 'contact=full_contacts;'),
      ('rows.append(row);failure={', "row['joint_limit_margins']=joint_margins(m,d,arm);rows.append(row);failure={"),
      ('   rows.append(row)\n', "   row['joint_limit_margins']=joint_margins(m,d,arm)\n   rows.append(row)\n"),
      ("report['initial_contacts']=initial", "report['initial_contacts']=initial;report['initial_joint_limit_margins']=joint_margins(m,d,arm);report['orientation_choice']='single bilateral 180degree jaw mirror; unchanged physical approach axis'")
    ]
    for a,b in edits:
        if src.count(a)!=1:raise ValueError('Exact scoped copied seam changed: '+a)
        src=src.replace(a,b)
    return src

def main():
    src=selected_source()
    if (H/'executed_source.py.txt').read_text()!=src:raise ValueError('Prepared compiled source mismatch')
    ns=dict(original.main.__globals__)
    ns.update(H=H,ROOT=ROOT,joint_margins=joint_margins,full_contacts=full_contacts,save_exact_constructor=save_exact_constructor)
    exec(compile(src,str(H/'check.py')+'::one-bilateral-mirror-copied-only','exec'),ns)
    ns['main']()
if __name__=='__main__':main()
