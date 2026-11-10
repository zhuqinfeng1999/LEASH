"""Explicit saved-model offline initial design and ONE continuous copied route."""
import inspect,textwrap,json,hashlib,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np,mujoco
from research.v4.candidates.r3_04_source_axis_mirrored_low_copy_20260928 import check as mirrored
from research.v4.candidates.r3_04_source_axis_low_entry_copy_20260928 import check as original
from research.v4.accelerated.grasp_assist import GraspAssist
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
SAVED=H.parent/'r3_04_source_axis_mirrored_low_copy_20260928'

def saved_offline_frame(seed):
    if seed!=0:raise ValueError('One exact saved own seed0 frame only')
    receipt=json.loads((SAVED/'constructor_snapshot.json').read_text())
    for name,sha in receipt['files'].items():
        if hashlib.sha256((SAVED/name).read_bytes()).hexdigest()!=sha:raise ValueError('Saved compiled frame drift '+name)
    m=mujoco.MjModel.from_binary_path(str(SAVED/'compiled_model.mjb'))
    snap=np.load(SAVED/'constructor_data.npz',allow_pickle=False)
    old=mujoco.MjData(m);mujoco.mj_setState(m,old,snap['integration_state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,old)
    center=m.site('gripper0_right_grip_site').id
    joint=m.joint('robot0_joint7').id;address=int(m.jnt_qposadr[joint]);qold=float(old.qpos[address])
    d=mujoco.MjData(m);mujoco.mj_setState(m,d,snap['integration_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    d.qpos[address]+=math.pi;mujoco.mj_forward(m,d)
    before=old.site_xmat[center].reshape(3,3);after=d.site_xmat[center].reshape(3,3)
    pdelta=float(np.linalg.norm(d.site_xpos[center]-old.site_xpos[center]));rotation_delta=float(np.max(np.abs(after-before@np.diag([-1.,-1.,1.]))))
    if pdelta>1e-12 or rotation_delta>1e-12:raise ValueError('Actual copied initial EEF/jaw symmetry FK failed')
    assert np.array_equal(np.delete(d.qpos,address),np.delete(old.qpos,address))
    ranges=m.jnt_range[joint];assert ranges[0]<d.qpos[address]<ranges[1]
    kwargs={'hand_body':'gripper0_right_eef','center_site':'gripper0_right_grip_site','left_pad_site':'v42_left_pad','right_pad_site':'v42_right_pad',
      'slot_metadata':{'L':{'grasp_zones':[{'object_body':'foil_sandwich_1_main','zone_site':'v42_foil_sandwich_1_grasp_zone','weld_name':'v42_foil_sandwich_1_assist_weld','tolerance_xyz':(.015,.015,.020)}]}}}
    assist=GraspAssist(m,d,**kwargs)
    initial={'source':'377 saved original own compiled binary/integration state; no Scene constructed','original_joint7':qold,'designed_initial_joint7':float(d.qpos[address]),'joint7_margins':[float(d.qpos[address]-ranges[0]),float(ranges[1]-d.qpos[address])],
      'EEF_position_delta_m':pdelta,'EEF_approach_axis_delta':float(np.max(np.abs(after[:,2]-before[:,2]))),'jaw_bilateral_rotation_matrix_max_error':rotation_delta,
      'old_EEF':old.site_xpos[center].tolist(),'new_EEF':d.site_xpos[center].tolist(),'old_R':before.tolist(),'new_R':after.tolist(),
      'all_other_qpos_exact':True,'private_offline_assist_metadata_from_actual_compiled_own_names':kwargs,'actual_physical_attachment_or_controls':0,'actual_Scene_constructors':0,'native_ready':False}
    (H/'initial_fk_receipt.json').write_text(json.dumps(initial,indent=2)+'\n')
    np.savez_compressed(H/'designed_initial_copied_qpos.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),eq_active=d.eq_active.copy())
    return SimpleNamespace(backend=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=m),data=SimpleNamespace(_data=d)),
          objects_dict={'foil_sandwich_1':SimpleNamespace(root_body='foil_sandwich_1_main',joints=['foil_sandwich_1_joint0']),
                        'microwave_1':SimpleNamespace(root_body='microwave_1_main')},grasp_assist_kwargs=kwargs),
          grasp_assist=assist,item_name='foil_sandwich_1',oven_name='microwave_1',close=lambda:None)

def no_restart_ik():
    _,src=original.instrumented_ik()
    old='for restart in (0, 1):'
    if src.count(old)!=1:raise ValueError('Original restart seam drift')
    src=src.replace(old,'for restart in (0,): # root requested first restart-needed STOP; no alternative qpos initialization')
    src=src.replace('"best_full_qpos": best_qpos.tolist()', '"restart_requested": 1, "restart_performed": False, "best_full_qpos": best_qpos.tolist()')
    ns=dict(original._ik.__globals__);exec(compile(src,str(H/'check.py')+'::same-first-IK-no-restart','exec'),ns)
    return ns['_ik'],src

def selected_source():
    src=mirrored.selected_source()
    old="scene=HorizontalSideFoilScene(0);report['Scene_CPU_constructors']=1;m=scene.backend.sim.model._model;report['exact_constructor_snapshot']=save_exact_constructor(scene,m);d=mujoco.MjData(m);"
    new="scene=saved_offline_frame(0);report['Scene_CPU_constructors']=0;report['saved_model_loads']=1;report['designed_initial_joint7_plus_pi']=True;m=scene.backend.sim.model._model;d=mujoco.MjData(m);"
    if src.count(old)!=1:raise ValueError('Single original constructor seam changed')
    src=src.replace(old,new)
    src=src.replace('ik,src=instrumented_ik();','ik,src=no_restart_ik();')
    return src

def main():
    src=selected_source()
    if (H/'executed_source.py.txt').read_text()!=src:raise ValueError('Prepared own source changed')
    ns=dict(original.main.__globals__)
    ns.update(H=H,ROOT=ROOT,joint_margins=mirrored.joint_margins,full_contacts=mirrored.full_contacts,saved_offline_frame=saved_offline_frame,no_restart_ik=no_restart_ik)
    exec(compile(src,str(H/'check.py')+'::one-saved-initial-mirror-copy','exec'),ns);ns['main']()
if __name__=='__main__':main()
