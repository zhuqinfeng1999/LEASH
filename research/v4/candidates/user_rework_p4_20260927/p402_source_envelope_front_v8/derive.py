"""One uniform source shift from continuous yaw/translation support, no Scene/IK."""
import hashlib,itertools,json,math
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.p4_02_straightener_heatmat_20260926.facade import REGIONS
H=Path(__file__).resolve().parent;P=H.parent/'p402_mat_front_v5'
C=np.array(list(itertools.product((-1,1),repeat=3)))
def trig_extreme(c,a,b,lo,hi,minimum=True):
    ts=[lo,hi];p=math.atan2(b,a)+(math.pi if minimum else 0)
    ts += [p+2*math.pi*k for k in(-1,0,1)if lo<=p+2*math.pi*k<=hi]
    vals=[c+a*math.cos(t)+b*math.sin(t)for t in ts];i=int(np.argmin(vals)if minimum else np.argmax(vals));return vals[i],ts[i]
def main():
    if(H/'derivation.json').exists():raise RuntimeError('Preserve sole envelope derivation')
    old=json.loads((P/'copied.json').read_text());touch=next(x for x in old['rows']if x.get('real_region_eligible'))
    m=mujoco.MjModel.from_xml_path(str(H.parent/'p402_front_vanity_v1/compiled.xml'));d=mujoco.MjData(m)
    for i,q in enumerate(touch['joint_qpos_rad'],1):d.qpos[m.jnt_qposadr[m.joint('robot0_joint'+str(i)).id]]=q
    for i,q in enumerate((.04,-.04),1):d.qpos[m.jnt_qposadr[m.joint('gripper0_right_finger_joint'+str(i)).id]]=q
    mujoco.mj_forward(m,d)
    sid=m.site('gripper0_right_grip_site').id;EEF=d.site_xpos[sid];R=d.site_xmat[sid].reshape(3,3)
    # Exact board-relative top-down frame: source tool is boardR@Rz180.
    B=np.array([[0.,-1.,0.],[-1.,0.,0.],[0.,0.,-1.]])
    cold=np.array(REGIONS[0].start)+(.005/.055)*(np.array(REGIONS[0].end)-np.array(REGIONS[0].start));body_to_eef=np.diag([-1.,-1.,1.])@cold
    hand=[]
    for g in range(m.ngeom):
        if not(m.geom(g).name or'').startswith('gripper0_'):continue
        Q=d.geom_xmat[g].reshape(3,3);center=d.geom_xpos[g];kind=int(m.geom_type[g])
        if kind==int(mujoco.mjtGeom.mjGEOM_BOX):v=C*m.geom_size[g]@Q.T+center
        elif kind==int(mujoco.mjtGeom.mjGEOM_MESH):
            mid=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);v=np.asarray(m.mesh_vert[a:a+n])@Q.T+center
        else:raise RuntimeError('Unexpected actual gripper geometry')
        local=(v-EEF)@R;board=local@B.T+body_to_eef
        i=int(np.argmin(board[:,1]));hand.append(dict(geom=m.geom(g).name,minimum_frontward_board_y_relative_tool_body_m=float(board[i,1]),active_compiled_vertex=i,actual_vertices=len(v),min_z_m=float(board[:,2].min()),max_z_m=float(board[:,2].max()),physical_collision=bool(m.geom_contype[g]or m.geom_conaffinity[g])))
    active=min(hand,key=lambda x:x['minimum_frontward_board_y_relative_tool_body_m']);hmin=active['minimum_frontward_board_y_relative_tool_body_m'];a=math.radians(10)
    # Source [-.15,-.22] minus board[-.08,.02], independent ±.02XY each.
    # Front normal=(-cos(theta),-sin(theta)), θ in±10deg.
    # Worst relative translation projection is -.04(cos(theta)+abs(sin(theta))).
    branches=[]
    for lo,hi,sign in[(-a,0.,-1),(0.,a,1)]:
        val,theta=trig_extreme(.13+hmin,.07-.04,.24-.04*sign,lo,hi,True)
        branches.append(dict(yaw_interval_rad=[lo,hi],minimum_existing_assembly_front_y_m=val,extremum_yaw_rad=theta))
    worst=min(branches,key=lambda x:x['minimum_existing_assembly_front_y_m']);newshift=max(0.,.12+.005-worst['minimum_existing_assembly_front_y_m'])
    # Same280x100 actual plinth, rotated byboardR@Rz180, source ±.02.
    # WorldminimumX = -.17-(.18+shift)cos(theta)-.14abs(sin(theta)).
    bounds=[]
    for lo,hi,sign in[(-a,0.,-1),(0.,a,1)]:
        x,t=trig_extreme(-.17,-(.18+newshift),-.14*sign,lo,hi,True)
        bounds.append(dict(yaw_interval_rad=[lo,hi],minimum_plinth_x_m=x,extremum_yaw_rad=t))
    floor=m.geom('table_collision').id;tablemin=float(d.geom_xpos[floor,0]-m.geom_size[floor,0]);minx=min(x['minimum_plinth_x_m']for x in bounds)
    # Direct uniformshift upper bound from exact endpoint constraints; no height/location sampling.
    limits=[]
    for t in(-a,0.,a):limits.append((- .17-.18*math.cos(t)-.14*abs(math.sin(t))-tablemin)/math.cos(t))
    max_support_shift=min(limits)
    r=dict(kind='continuous full-open-hand source/vanity plane support derivation',finger_qpos_m=[.04,-.04],source_tool_yaw_is_board_yaw_plus180deg=True,public_seed_domain=dict(source_and_board_xy_each_plus_minus_m=.02,board_yaw_plus_minus_deg=10),continuous_support_branches=branches,whole_hand_real_visual_and_collision_minima=hand,active_hand_geometry=active,existing_source_shift_m=.13,minimum_extra_common_front_shift_m=newshift,new_total_common_front_shift_m=.13+newshift,required_front_plane_m=.125,after_shift_worst_front_plane_m=worst['minimum_existing_assembly_front_y_m']+newshift,plinth_dimensions_m=[.28,.10,.032],plinth_worldx_bounds= bounds,worst_plinth_minimum_x_m=minx,actual_table_minimum_x_m=tablemin,maximum_extra_shift_with_entire_plinth_supported_m=max_support_shift,required_shift_exceeds_full_support_by_m=newshift-max_support_shift,full_plinth_support_possible=bool(minx>=tablemin),nominal_new_source_xyz=[-.15-(.13+newshift),-.22,.939],minimum_tabletop_y_source_envelope_m=-.24-(.13+newshift)*math.sin(a)-.14*math.cos(a)-.05*math.sin(a),Scene=0,IK=0,controls=0,physics_steps=0,GPU_models=0,limitation='Uniform front-plane separation is necessary sufficient for full open hand clearing vanity; no pose/position grid. If source plinth leaves table, no candidate is implemented or copied route attempted.')
    (H/'derivation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in('active_hand_geometry','minimum_extra_common_front_shift_m','worst_plinth_minimum_x_m','actual_table_minimum_x_m','maximum_extra_shift_with_entire_plinth_supported_m','required_shift_exceeds_full_support_by_m','full_plinth_support_possible')}))
if __name__=='__main__':main()
