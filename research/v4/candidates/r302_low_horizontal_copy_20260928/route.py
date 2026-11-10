"""One root-approved low carry copied diagnostic; no Scene on import."""
import inspect,json
from pathlib import Path
from types import FunctionType
from research.v4.candidates.r302_horizontal_source_geometry_v2_20260928 import route as parent
from research.v4.accelerated.slot_preflight import _ik as original_ik
H=Path(__file__).resolve().parent
PROPOSAL=H.parent/'r302_low_horizontal_carry_proposal_20260928/proposal.json'
PADS=frozenset(('gripper0_right_finger1_pad_collision','gripper0_right_finger2_pad_collision'))
PAD_SEAM="finger_grasp=(a.startswith('gripper0_')and 'finger'in a and pay(b))or(b.startswith('gripper0_')and'finger'in b and pay(a))"
def pad_contact_helper(original):
    src=inspect.getsource(original)
    if src.count(PAD_SEAM)!=1 or src.count("'fork_bowl_1_'")!=1:raise RuntimeError('Exact contact source drift')
    revised=src.replace("'fork_bowl_1_'","'steel_thermos_1_'").replace(PAD_SEAM,"finger_grasp=(a in PADS and pay(b))or(b in PADS and pay(a))")
    ns=dict(original.__globals__,PADS=PADS)
    exec(compile(revised,str(H/'route.py')+'::only-actual-pads-exempt','exec'),ns)
    return ns['contact_rows']

def recorded_ik():
    src=inspect.getsource(original_ik)
    old='best = (pos_error + .06 * rot_error, pos_error, rot_error, iteration)'
    ret='"reason": "ik_no_convergence_or_joint_limit"}'
    if src.count(old)!=1 or src.count(ret)!=1:raise RuntimeError('Original IK diagnostic seam drift')
    src=src.replace(old,old+'\n                best_full_qpos = data.qpos.copy(); best_restart = restart')
    src=src.replace(ret,'"reason": "ik_no_convergence_or_joint_limit", "best_full_qpos": best_full_qpos.tolist(), "best_restart": best_restart}')
    ns=dict(original_ik.__globals__);exec(compile(src,str(H/'route.py')+'::same-IK-best-state-only','exec'),ns)
    return ns['_ik']

def selected_main():
    _,original,src=parent.selected_main()
    lift='lift=eef.copy();lift[2]=1.23'
    align='aligned=op+oR@bLocal-R@rel[0]'
    target='target=op+oR@tl-R@rel[0]'
    badik="if not ik['reachable']:rows.append(row);failure=dict(phase=phase,kind='original copied IK not reachable',detail=ik);return False"
    for needle in ('contacts=bind_contact_helper(original_contacts)',lift,align,target,badik):
        if src.count(needle)!=1:raise RuntimeError('Own route source seam drift: '+needle)
    src=src.replace('contacts=bind_contact_helper(original_contacts)','contacts=pad_contact_helper(original_contacts)\n for name in PADS:\n  model.geom(name).id # Actual compiled exact-name admission, no generic finger exemption.\n initialview=SimpleNamespace(backend=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=model),data=SimpleNamespace(_data=data))))\n initial_contact_rows=contacts(initialview);initial_qpos=data.qpos.copy()')
    src=src.replace(lift,'lift=eef.copy();lift[2]=LOW_LEVEL')
    src=src.replace(align,align+';aligned[2]=LOW_LEVEL')
    src=src.replace(target,target+';target[2]=LOW_LEVEL')
    src=src.replace(badik,"""if not ik['reachable']:
   data.qpos[:]=ik['best_full_qpos'];mujoco.mj_forward(model,data)
   if held:
    bestE=data.site_xmat[center].reshape(3,3);besteef=data.site_xpos[center].copy()
    data.qpos[adr:adr+3]=besteef+bestE@rel[0];data.qpos[adr+3:adr+7]=Rotation.from_matrix(bestE@rel[1]).as_quat()[[3,0,1,2]];mujoco.mj_forward(model,data)
   bestview=SimpleNamespace(backend=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=model),data=SimpleNamespace(_data=data))))
   row.update(actualEEF=data.site_xpos[center].tolist(),qpos=data.qpos.tolist(),contacts=contacts(bestview),diagnostic_best_only_not_reached=True)
   rows.append(row);failure=dict(phase=phase,kind='original copied IK not reachable',detail=ik);return False""")
    close="report['closure_eligibility']=dict(actual_zone_local=local.tolist(),along=along,gap=float(np.linalg.norm(v)),same_frozen_tol=[.015,.015,.020])"
    if src.count(close)!=1:raise RuntimeError('Closure record seam drift')
    src=src.replace(close,close+"""
    cv=SimpleNamespace(backend=SimpleNamespace(sim=SimpleNamespace(model=SimpleNamespace(_model=model),data=SimpleNamespace(_data=data))))
    cc=contacts(cv);rows.append(dict(phase='actual_legal_copied_closure',t=1.,qpos=data.qpos.tolist(),actualEEF=eef.tolist(),contacts=cc))
    if any(c['non_grasp_hard_stop']for c in cc):
     failure=dict(phase='actual_legal_copied_closure',kind='Actual hard finger/palm/arm/env overlap>1mm; only pads exempt',contacts=[c for c in cc if c['non_grasp_hard_stop']]);raise RuntimeError('First copied closure hard overlap; STOP')""")
    src=src.replace("except Exception as e:failure=dict(phase='instrumentation',kind=type(e).__name__,error=str(e))","except Exception as e:\n  if failure is None:failure=dict(phase='instrumentation',kind=type(e).__name__,error=str(e))")
    report='report=dict(constructors=1,Scene_controls=0,GPU=False,models=0,source_unchanged=True,copied_data_only=True)'
    if src.count(report)!=1:raise RuntimeError('Report source drift')
    src=src.replace(report,report+"\n report.update(initial_qpos=initial_qpos.tolist(),initial_contacts=initial_contact_rows,pad_exempt_names=sorted(PADS),unique_low_level=LOW_LEVEL)")
    ns=dict(parent.old.main.__globals__,H=H,ORIGINAL_H=parent.ORIGINAL_H,pad_contact_helper=pad_contact_helper,PADS=PADS,LOW_LEVEL=float(json.loads(PROPOSAL.read_text())['carry_EEF_z_m']),_ik=recorded_ik())
    exec(compile(src,str(H/'route.py')+'::one-low-level-pad-only-copy','exec'),ns)
    return ns['main'],src
if __name__=='__main__':selected_main()[0]()
