"""Prospective private top sequence on original OSC; no task execution here."""
import inspect
from pathlib import Path

from research.v4.candidates.r305_tall_microwave_inside12_20260928 import expert as parent
from research.v4.candidates.r305_robot_facing_lug_y80_copy_20260928.actual_expert import initial, journal
from .check import clearance_waypoint


def selected_naive():
    source=inspect.getsource(parent.run_naive)
    old=("    above=target.copy();above[2]=1.15\n"
         "    pose(above,R,'radial_above_original_bowl',max_steps=180)\n"
         "    pose(target,R,'radial_vertical_material_approach',max_steps=160,clip=.006)")
    new=("    details['private_expert_grip_version']='standard-top-original20mm-ear-v1'\n"
         "    home,above=clearance_waypoint(m,d,center,R,target,details)\n"
         "    pose(home,R,'standard_top_orient_at_existing_home',max_steps=180)\n"
         "    pose(above,R,'standard_top_lateral_above_material',max_steps=180)\n"
         "    pose(target,R,'standard_top_vertical_material_approach',max_steps=160,clip=.006,\n"
         "        tol=.001,rotation_tol=np.deg2rad(1))")
    changes={old:new,
        'carry=max(float(op[2]+lower),float(origin[2]+.020))':
        'carry=max(float(op[2]+lower),float(origin[2]+.032))',
        "    details['actual_private_lug_design']=design":
        "    details['actual_private_lug_design']=design\n"
        "    details['private_top_control_version']={'original_OSC_intermediate_tol_m':.008,\n"
        "        'copied_default_position_tol_m':.012,'material_position_tol_m':.001,\n"
        "        'material_rotation_tol_deg':1,'requested_lift_waypoint_m':.032,\n"
        "        'original_lift_qualification_m':.020,'original_total_budget':[1200,90.],\n"
        "        'no_state_relative_qpos_or_controller_write':True}"}
    for before,after in changes.items():
        if source.count(before)!=1:raise RuntimeError('Original top actual expert seam differs '+before)
        source=source.replace(before,after)
    namespace=dict(parent.run_naive.__globals__,clearance_waypoint=clearance_waypoint)
    exec(compile(source,str(Path(__file__).resolve())+'::original-own-OSC-top-sequence','exec'),namespace)
    fn=namespace['run_naive']
    for name in ('generic_driver_source','lug_material'):
        if fn.__globals__[name]is not parent.run_naive.__globals__[name]:
            raise RuntimeError('Original actual driver/material helper replaced '+name)
    return fn,source


run_naive,SELECTED_NAIVE_SOURCE=selected_naive()
