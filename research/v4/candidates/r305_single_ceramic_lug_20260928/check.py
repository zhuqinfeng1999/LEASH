"""One real-lug material-frame geometry observation over an independent saved model."""
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType
import textwrap

import mujoco
import numpy as np

from research.v4.candidates.r305_radial_rim_saved_copy_20260928 import check as parent

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
SAVED = H.parent / 'r305_front_source_material_copy_20260928'
ORIGINAL_IK = parent._ik


def lug_material(m, d, center, body, left, right, finger):
    zone=m.site('v42_fork_bowl_1_grasp_zone').id
    lug=m.geom('fork_bowl_1_ceramic_lug_collision').id
    if m.site_pos[zone].tolist()!=[.080,0.,.022]:raise RuntimeError('Exact new ceramic-lug annotation changed')
    Q0=d.site_xmat[center].reshape(3,3);p0=d.site_xpos[center].copy()
    lp=Q0.T@(d.site_xpos[left]-p0);rp=Q0.T@(d.site_xpos[right]-p0)
    jaw_axis=(rp-lp)/np.linalg.norm(rp-lp)
    if not np.allclose(jaw_axis,[1.,0.,0.],rtol=0,atol=1e-10):raise RuntimeError('Original jaw axis changed')
    B=d.xmat[body].reshape(3,3)
    tangent=B[:,1].copy();tangent[2]=0.;tangent/=np.linalg.norm(tangent)
    down=np.array([0.,0.,-1.]);R=np.column_stack((tangent,np.cross(down,tangent),down))
    material=d.site_xpos[zone].copy()
    negative=float(mujoco.mj_rayMesh(m,d,lug,material,-tangent))
    positive=float(mujoco.mj_rayMesh(m,d,lug,material,tangent))
    if not (negative>0 and positive>0):raise RuntimeError('New grasp annotation is not inside actual solid lug')
    width=negative+positive
    if not np.isclose(width,.020,rtol=0,atol=1e-5):raise RuntimeError('Actual lug tangential material width differs from20mm')
    material=material+(positive-negative)/2*tangent
    midpoint=(lp+rp)/2
    target=material-R@midpoint
    pad=m.geom(parent.PADS[0]).id
    half=float(np.sum(np.abs((Q0.T@d.geom_xmat[pad].reshape(3,3))[0])*m.geom_size[pad]))
    offsets=[abs(float(lp[0]))-float(d.qpos[m.jnt_qposadr[finger[0]]]),
             abs(float(rp[0]))-abs(float(d.qpos[m.jnt_qposadr[finger[1]]]))]
    closed_q=width/2+half-sum(offsets)/2
    if not 0.<closed_q<=.04:raise RuntimeError('New actual material/pad closure not within legal finger range')
    design={'version':'R305-one-ear-32x20x10-worlddown-tangent-pinch-v1',
        'grasp_region_is_visible_lug_center':True,'original_zone_local':m.site_pos[zone].tolist(),
        'actual_zone_world':d.site_xpos[zone].tolist(),'actual_material_mesh':m.geom(lug).name,
        'negative_material_ray_m':negative,'positive_material_ray_m':positive,'material_width_m':width,
        'jaw_local_axis':jaw_axis.tolist(),'jaw_world_tangent':tangent.tolist(),'approach_world':down.tolist(),
        'actual_pad_midpoint_EEF_local':midpoint.tolist(),'actual_pad_half_width':half,
        'pad_zero_offsets':offsets,'closed_finger_qpos':[closed_q,-closed_q],
        'material_center':material.tolist(),'target_EEF':target.tolist(),'target_R':R.tolist(),
        'no_q0_closure':True,'no_old_physical_qualification_inherited':True}
    return R,target,closed_q,design


def logged_ik(proxy, target, R, **kwargs):
    answer=ORIGINAL_IK(proxy,target,R,**kwargs)
    p=H/'first_IK_return.json'
    if not p.exists():
        p.write_text(json.dumps({'actual_return':answer,'target':np.asarray(target).tolist(),
            'R':np.asarray(R).tolist(),'parameters':kwargs,'actual_callable':ORIGINAL_IK.__module__+'.'+ORIGINAL_IK.__name__},indent=2)+'\n')
    return answer


def selected():
    src=textwrap.dedent(inspect.getsource(parent.main))
    edits=[("m = mujoco.MjModel.from_binary_path(str(SAVED / 'actual_model.mjb'))",
            "m = mujoco.MjModel.from_binary_path(str(H / 'lug_model.mjb'))"),
           ("m.site_pos[zone].tolist() != [.059,0.,.022]","m.site_pos[zone].tolist() != [.080,0.,.022]"),
           ("raise RuntimeError('Original59mm zone/310g shape source changed')",
            "raise RuntimeError('New visible80mm lug zone/310g source changed')")]
    for old,new in edits:
        if src.count(old)!=1:raise RuntimeError('Original geometry helper seam changed: '+old)
        src=src.replace(old,new)
    start='        p0 = d.site_xpos[center].copy(); Q0 = d.site_xmat[center].reshape(3,3).copy()\n'
    end='        proxy = SimpleNamespace(model=SimpleNamespace(_model=m),data=SimpleNamespace(qpos=d.qpos))\n'
    if src.count(start)!=1 or src.count(end)!=1:raise RuntimeError('Original material definition boundary changed')
    a=src.index(start);b=src.index(end,a)
    src=src[:a]+"        R, target, closed_q, result['design'] = lug_material(m, d, center, body, left, right, finger)\n"+src[b:]
    ns=dict(parent.main.__globals__);ns.update(H=H,ROOT=ROOT,SAVED=SAVED,lug_material=lug_material,_ik=logged_ik)
    exec(compile(src,__file__+'::actual-original-two-frames-and-continuous-copy-new-lug-material','exec'),ns)
    return ns['main'],src


if __name__=='__main__':selected()[0]()
