"""One exact full-can floor/footprint, no scene, sampling or physics."""
import math
from pathlib import Path

import mujoco
import numpy as np

from research.v4.candidates.r302_horizontal_end_grasp_saved_copy_20260928 import derive
from research.v4.candidates.r302_horizontal_short_v_20260928.copy import shape_interval

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
RAW = ROOT/'research/v4/candidates/r302_source_clearance_ten_seed_prepare_20260928/root_qualification_seed0_9/correct_seed2'
OLD_XML = ROOT/'research/v4/candidates/r302_condim6_top_mirror_tall350_saved_copy_20260928/compiled_seed0/scene.xml'
CAN_NAMES = derive.CAN_NAMES
WOOD_NAMES = ('r302_v_base', 'r302_v_rail_-1', 'r302_v_rail_1',
              'r302_ledge_end_-1', 'r302_ledge_end_1')
CLEARANCE = .005
FENCE_THICKNESS = .003
FENCE_HEIGHT = .006
TABLE_Z = .9


def derive_layout():
    m = mujoco.MjModel.from_binary_path(str(RAW/'startup_seed0/compiled_model.mjb'))
    raw = np.load(RAW/'initial_constructor_state.npz', allow_pickle=False)
    d = mujoco.MjData(m); d.qpos[:] = raw['qpos']; mujoco.mj_forward(m,d)
    body=m.body('steel_thermos_1_main').id; wood=m.body('r302_v_wood_cradle').id
    cans=[m.geom(n).id for n in CAN_NAMES]
    if abs(float(m.body_mass[body])-.4)>1e-12 or not np.array_equal(m.site_pos[m.site('v42_steel_thermos_1_grasp_zone').id],[0,0,.042]):
        raise RuntimeError('Actual original400g/+42mm region mismatch')
    B=d.xmat[wood].reshape(3,3); p=d.xpos[wood]
    intervals=[shape_interval(m,d,cans,B[:,i],p) for i in range(3)]
    if not np.allclose(intervals[:2],[[-.0575,.0575],[-.0365,.0365]],rtol=0,atol=1e-12):
        raise RuntimeError('Complete actual73x115mm can footprint is not the declared one')
    floor=float(intervals[2][0]); inner=np.asarray([max(abs(v) for v in q)+CLEARANCE for q in intervals[:2]])
    outer=inner+FENCE_THICKNESS
    if not .010 < floor < .050:
        raise RuntimeError('Derived table-backed wood thickness is not a normal source ledge')
    # Exact extrema over continuous common yaw +/-10deg, not a coordinate grid.
    yaw_half=math.radians(10)
    def half_envelope(a,b):
        theta=math.atan2(b,a)
        angles=[0.,yaw_half]
        if theta<=yaw_half:angles.append(theta)
        return max(a*math.cos(t)+b*math.sin(t) for t in angles)
    half=np.asarray([half_envelope(outer[0],outer[1]),half_envelope(outer[1],outer[0])])
    nominal=np.asarray([-.08,-.312266002685539])
    envelope=[(nominal-half-.020).tolist(),(nominal+half+.020).tolist()]
    table_clearance=float(np.min(.45-np.abs(np.asarray(envelope))))
    if table_clearance<.005:raise RuntimeError('Full perturbed wooden ledge is not truly on the unchanged tabletop')
    return dict(floor_top_local_m=floor,floor_top_world_m=TABLE_Z+floor,
        inner_half_footprint_m=inner.tolist(),outer_half_footprint_m=outer.tolist(),
        outside_dimensions_m=[2*outer[0],2*outer[1],floor+FENCE_HEIGHT],
        fence_thickness_m=FENCE_THICKNESS,fence_height_m=FENCE_HEIGHT,
        full_actual_can_intervals_in_wood_m=[list(q) for q in intervals],
        complete_source_yaw_xy_footprint_world_m=envelope,continuous_table_clearance_m=table_clearance,
        can_center_original_world_m=d.xpos[body].tolist(),actual_wood_pos=m.body_pos[wood].tolist(),
        actual_wood_quat=m.body_quat[wood].tolist(),can_mass_kg=float(m.body_mass[body]),
        no_can_source_center_change=True,no_preload=True,wood_is_fixed_table_supported=True,
        gravity_support_points_local_m=[[-.05575,0,floor],[.05575,0,floor]],
        support_normals_world=[[0,0,1],[0,0,1]],
        CoM_projection_in_support_segment=True,
        rolling_rest_not_proven=True,requires_actual_fixed80_rest_witness=True)


LAYOUT=derive_layout()
