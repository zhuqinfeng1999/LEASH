"""Old64 saved poses first; then exactly one original continuous copied route."""
import hashlib
import json
from pathlib import Path
from types import FunctionType
import time

import mujoco
import numpy as np

from research.v4.candidates.r305_single_ceramic_lug_20260928 import check as positive
from research.v4.candidates.r305_radial_rim_saved_copy_20260928 import check as radial

H=Path(__file__).resolve().parent
ROOT=H.parents[3]
P=H.parent/'r305_single_ceramic_lug_20260928'
ORIGINAL_IK=positive.ORIGINAL_IK


def full_contacts(m,d):
    rows=radial.contact_rows(m,d)
    for row in rows:
        names=row['geoms']
        if (any(n.startswith('fork_bowl_1_')for n in names)
            and any(n.startswith('microwave_1_')for n in names)
            and row['dist_m']<-.001):
            row['hard_stop']=True
            row['payload_device_hard_contact']=True
    return rows


def logged_ik(proxy,target,R,**kwargs):
    answer=ORIGINAL_IK(proxy,target,R,**kwargs)
    path=H/'first_IK_return.json'
    if not path.exists():
        path.write_text(json.dumps({'actual_return':answer,'target':np.asarray(target).tolist(),
                                   'R':np.asarray(R).tolist(),'parameters':kwargs,
                                   'unchanged_actual_callable':ORIGINAL_IK.__module__+'.'+ORIGINAL_IK.__name__},indent=2)+'\n')
    return answer


def selected():
    original,source=positive.selected()
    ns=dict(original.__globals__)
    ns.update(H=H,ROOT=ROOT,_ik=logged_ik,contact_rows=full_contacts)
    selected=FunctionType(original.__code__,ns,original.__name__,original.__defaults__,original.__closure__)
    if selected.__code__ is not original.__code__:raise RuntimeError('Original copied-route code changed')
    return selected,source


def fixed64():
    if (H/'fixed64_report.json').exists():raise RuntimeError('One fixed64 observation already saved')
    pins=json.loads((H/'source_freeze.json').read_text())
    for name,expected in pins.items():
        path=Path(name)if Path(name).is_absolute()else ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Source drift '+name)
    old=json.loads((P/'report.json').read_text())
    if len(old['route'])!=64:raise RuntimeError('Original saved64 evidence changed')
    out={'passed':False,'failure':None,'rows':[],'source_sha256':pins,'Scene_constructors':0,
         'actual_controls':0,'IK_calls':0,'GPU_models_services':0,
         'evidence_scope':'fixed saved qpos contact evaluation in independent new model; not dynamic control'}
    start=time.monotonic()
    try:
        build=json.loads((H/'build_report.json').read_text())
        if not build['passed']:raise RuntimeError('New device compile proof unavailable')
        m=mujoco.MjModel.from_binary_path(str(H/'lug_model.mjb'));d=mujoco.MjData(m)
        eef=m.site('gripper0_right_grip_site').id
        for i,row in enumerate(old['route']):
            d.qpos[:]=row['qpos'];mujoco.mj_forward(m,d)
            cs=full_contacts(m,d)
            out['rows'].append({'index':i,'phase':row['phase'],'qpos':d.qpos.tolist(),
                                'EEF':d.site_xpos[eef].tolist(),'contacts':cs,
                                'saved_EEF_exact':bool(np.array_equal(d.site_xpos[eef],row['EEF']))})
            if not out['rows'][-1]['saved_EEF_exact']:raise RuntimeError('Unintended saved robot FK change')
            if any(c['hard_stop']for c in cs):raise RuntimeError('First fixed saved-pose hard contact at row '+str(i))
        out['passed']=True
    except Exception as exc:out['failure']=repr(exc)
    out['elapsed_s']=time.monotonic()-start
    (H/'fixed64_report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'fixed64_passed':out['passed'],'failure':out['failure'],'rows':len(out['rows']),'elapsed_s':out['elapsed_s']},indent=2))
    return out['passed']


def main():
    if fixed64():
        selected()[0]()


if __name__=='__main__':main()
