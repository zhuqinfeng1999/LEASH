"""Thin private side-frame configuration of the original actual OSC expert."""
import inspect
import json
from types import FunctionType

import mujoco
import numpy as np

from research.v4.candidates.r305_tall_microwave_inside12_20260928 import expert as parent
from research.v4.candidates.r305_horizontal_lug_y80_static_20260928.check import side_material
from research.v4.candidates.r305_tangent_material_saved_copy_20260928.check import horizontal_material_pre

initial=parent.initial


def outside_pre(m,d,center,R,target,details):
    # Reuse the exact geometry helper. Its single offline artifact is retained
    # in this case's private details instead of overwrite a pinned old file.
    class Receipt:
        def __truediv__(self,name):
            if name!='material_pre_design.json':raise RuntimeError('Original pre receipt seam differs')
            return self
        def write_text(self,text):
            details['actual_private_side_pre']=json.loads(text)
    ns=dict(horizontal_material_pre.__globals__);ns['H']=Receipt()
    fn=FunctionType(horizontal_material_pre.__code__,ns,horizontal_material_pre.__name__,
                    horizontal_material_pre.__defaults__,horizontal_material_pre.__closure__)
    return fn(m,d,center,R,target)


def journal(scene,action,evaluator,details):
    try:
        parent.journal(scene,action,evaluator,details)
    finally:
        # Only observations of the actual controller step; no forward/restore.
        if details.get('actual_control_rows'):
            m,d=scene.backend.sim.model._model,scene.backend.sim.data._data
            row=details['actual_control_rows'][-1];forces=[]
            for i,c in enumerate(d.contact[:d.ncon]):
                f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
                forces.append(dict(geoms=[m.geom(int(g)).name for g in (c.geom1,c.geom2)],
                    dist_m=float(c.dist),local_force=f.tolist(),world_force_on_geom2=(c.frame.reshape(3,3).T@f[:3]).tolist()))
            row['actual_contact_forces']=forces
            if scene.grasp_assist.held and 'actual_first_bind'not in details:
                _,b,_,w=scene.grasp_assist._entries[scene.grasp_assist._held_index]
                details['actual_first_bind']=dict(sequence=scene.sequence,sim_time_s=float(d.time),
                    actual_weld_name=m.equality(w).name,actual_weld_data=m.eq_data[w].tolist(),
                    actual_payload=d.xpos[b].tolist(),actual_EEF=d.site_xpos[scene.grasp_assist._center].tolist(),
                    no_saved_relation_or_qpos_installed=True)


def selected_naive():
    src=inspect.getsource(parent.run_naive)
    edits=[
        ('R,target,closed_q,design=lug_material(m,d,center,body,left,right,finger)',
         'R,target,closed_q,design=side_material(m,d,center,body,left,right,finger)'),
        ("    above=target.copy();above[2]=1.15\n    pose(above,R,'radial_above_original_bowl',max_steps=180)\n    pose(target,R,'radial_vertical_material_approach',max_steps=160,clip=.006)",
         "    pre=outside_pre(m,d,center,R,target,details)\n"
         "    above=pre.copy();above[2]=1.15\n"
         "    pose(above,R,'side_above_outside_robot_facing_ear',max_steps=180)\n"
         "    pose(pre,R,'side_lower_outside_robot_facing_ear',max_steps=160,clip=.006)\n"
         "    pose(target,R,'side_horizontal_material_approach',max_steps=160,clip=.006)"),
        ("    if not scene.grasp_assist.held:raise RuntimeError('Actual common closure did not bind')",
         "    details['actual_material_width_closure']={'derived_expected_qpos':[closed_q,-closed_q],\n"
         "        'actual_finger_qpos':[float(d.qpos[m.jnt_qposadr[j]])for j in finger],\n"
         "        'binary_original_gripper_controls':24,'no_finger_qpos_or_servo_state_write':True}\n"
         "    if not scene.grasp_assist.held:raise RuntimeError('Actual common closure did not bind')")]
    for old,new in edits:
        if src.count(old)!=1:raise RuntimeError('Original actual expert seam differs '+old)
        src=src.replace(old,new)
    ns=dict(parent.run_naive.__globals__,side_material=side_material,outside_pre=outside_pre)
    exec(compile(src,__file__+'::original-OSC-side-material-configuration','exec'),ns)
    return ns['run_naive'],src


run_naive,SELECTED_NAIVE_SOURCE=selected_naive()
