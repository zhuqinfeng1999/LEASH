"""One root-selected whole-source shift; copied geometry, never physical controls."""
import copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
from research.v4.candidates.r4_04_analytical_supported_layout_20260927.scene import Scene
from research.v4.candidates.r4_04_side_pinch_clearance_v2_20260927.derive import geom_points
from research.v4.candidates.r4_04_corner_material_geometry_20260927.readonly import cts
from research.v4.candidates.r4_03_neck_layout_readonly_20260927.check_geometry import state,geometry
from research.v4.candidates.r4_04_real_corner_zone_v2_20260926.project_cpu import screen
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
POINT=np.array([-.296,0.,.055]);SHIFT=.080
def digest_model(m):
 return {k:hashlib.sha256(getattr(m,k).tobytes()).hexdigest() for k in ('geom_pos','geom_size','geom_type','geom_contype','geom_conaffinity','mesh_vert','site_pos','body_mass','body_inertia','cam_pos','cam_quat','cam_fovy')}
def copy_source(s):
 om,od=s.backend.sim.model._model,s.backend.sim.data._data;m=copy.deepcopy(om);d=mujoco.MjData(m);mujoco.mj_setState(m,d,state(om,od),mujoco.mjtState.mjSTATE_INTEGRATION)
 obj=s.backend.objects_dict[s.item_name];jid=m.joint(obj.joints[0]).id;adr=m.jnt_qposadr[jid];d.qpos[adr]+=SHIFT;mujoco.mj_forward(m,d)
 return m,d,obj,jid,adr
def initial(s,m,d,obj):
 points=np.vstack([geom_points(m,d,m.geom(n).id)for n in obj.contact_geoms]);table=m.geom('table_collision').id;lo=d.geom_xpos[table,:2]-m.geom_size[table,:2];hi=d.geom_xpos[table,:2]+m.geom_size[table,:2];bounds=np.array([points.min(0),points.max(0)])
 support=bool(np.all(bounds[0,:2]>=lo)and np.all(bounds[1,:2]<=hi))
 contacts=[];bad=[]
 for c in d.contact[:d.ncon]:
  names=[m.geom(int(g)).name or''for g in(c.geom1,c.geom2)];pillow=any(n.startswith('adult_pillow_1_')for n in names);robot=any(n.startswith(('robot0_','gripper0_'))for n in names);person=any(n.startswith(('person_','human_','seated_person_'))for n in names)
  row=dict(pair=names,distance_m=float(c.dist),world_pos_m=c.pos.tolist());contacts.append(row)
  # Ordinary supported table contact is recorded separately, not declared
  # an unsafe deep overlap solely because of solver-scale compression.
  if c.dist<0 and (robot or person or (pillow and 'table_collision' not in names)):bad.append(row)
  if pillow and 'table_collision'in names and c.dist<-.001:bad.append(row)
 proxy=SimpleNamespace(model=s.backend.sim.model,data=SimpleNamespace(_data=d,cam_xpos=d.cam_xpos,cam_xmat=d.cam_xmat))
 projected=screen(proxy)
 return dict(full_collision_bounds_m=bounds.tolist(),table_xy_bounds_m=[lo.tolist(),hi.tolist()],whole_footprint_supported=support,contacts=contacts,blocking_initial_contacts=bad,public_projection=projected,passed=support and not bad and projected['screen_pass'])
def route(s,m,d,obj,jid,adr):
 g=s.grasp_assist;site=g._center;zone=m.site('v42_adult_pillow_1_grasp_zone').id;bid=m.body(obj.root_body).id;m.site_pos[zone]=POINT;mujoco.mj_forward(m,d)
 verts=geom_points(m,d,m.geom('adult_pillow_1_pillow_body_collision').id);vb=(d.xmat[bid].reshape(3,3).T@(verts-d.xpos[bid]).T).T;hull=ConvexHull(vb);margin=-(hull.equations[:,:3]@POINT+hull.equations[:,3]);lo=max(-(a*POINT[0]+b*POINT[1]+k)/c for a,b,c,k in hull.equations if c< -1e-7);hi=min(-(a*POINT[0]+b*POINT[1]+k)/c for a,b,c,k in hull.equations if c>1e-7)
 report=dict(material=dict(body_point_m=POINT.tolist(),inside_true_compiled_material=bool(np.all(margin>=-1e-7)),thickness_m=hi-lo,z_interval_m=[lo,hi],minimum_clearance_m=float(margin.min())),frames=[],error=None)
 if not report['material']['inside_true_compiled_material']:report.update(passed=False,error='authored seam outside actual material');return report
 arm=[m.joint(f'robot0_joint{i}').id for i in range(1,8)];adrs=[m.jnt_qposadr[i]for i in arm]
 for i in range(m.njnt):
  name=m.joint(i).name or''
  if 'finger_joint'in name:d.qpos[m.jnt_qposadr[i]]=m.jnt_range[i,1]if name.endswith('joint1')else m.jnt_range[i,0]
 mujoco.mj_forward(m,d);initial=d.site_xpos[site].copy();initialR=d.site_xmat[site].reshape(3,3).copy();q=d.qpos[adrs].copy();old=json.loads((ROOT/'research/v4/candidates/r4_04_side_pinch_clearance_v2_20260927/proposal.json').read_text());targetR=d.xmat[bid].reshape(3,3)@Rotation.from_euler('z',-np.pi/2).as_matrix()@np.asarray(old['SIDE_R']);target=d.site_xpos[zone].copy();local=[]
 for gi in range(m.ngeom):
  if(m.geom(gi).name or'').startswith('gripper0_')and m.geom_group[gi]==0:local.extend((initialR.T@(geom_points(m,d,gi)-initial).T).T)
 axis=targetR[:,2];allsource=np.vstack([geom_points(m,d,m.geom(n).id)for n in obj.contact_geoms]);extent=float(np.max((np.asarray(local)@targetR.T)@axis));dist=max(.15,float(target@axis-np.min(allsource@axis)+extent+.005));pre=target-axis*dist;above=np.array([pre[0],pre[1],max(1.2,pre[2])]);report['one_pose']=dict(targetR=targetR.tolist(),target_world_m=target.tolist(),prepoint_world_m=pre.tolist(),approach_distance_m=dist)
 way=[];interp=Slerp([0,1],Rotation.from_matrix([initialR,targetR]))
 for t in np.linspace(0,1,7)[1:]:way.append(('orient_at_reset',initial,interp([t]).as_matrix()[0]))
 for a,b,name,n in((initial,above,'above_outside_left_seam',10),(above,pre,'lower_outside_left_seam',8),(pre,target,'normal_side_entry_to_seam',12)):
  for t in np.linspace(0,1,n+1)[1:]:way.append((name,a+t*(b-a),targetR))
 def err(qp,xyz,R):
  d.qpos[adrs]=qp;mujoco.mj_forward(m,d);return np.r_[d.site_xpos[site]-xyz,.08*Rotation.from_matrix(R@d.site_xmat[site].reshape(3,3).T).as_rotvec()]
 for num,(phase,xyz,R)in enumerate(way):
  opt=least_squares(err,q,args=(xyz,R),bounds=(m.jnt_range[arm,0]+1e-6,m.jnt_range[arm,1]-1e-6),max_nfev=120,xtol=1e-9,ftol=1e-9,gtol=1e-9);q=opt.x;res=err(q,xyz,R);contact=cts(m,d);bad=[c for c in contact if c['distance_m']<0 and c['robot']and(c['pillow']or c['table']or c['infant']or any(x.startswith('fixed_mount')for x in c['pair'])or sum(x.startswith(('robot0_','gripper0_'))for x in c['pair'])==2)]
  row=dict(sequence=num,phase=phase,goal_xyz_m=xyz.tolist(),eef_xyz_m=d.site_xpos[site].tolist(),position_residual_m=float(np.linalg.norm(res[:3])),rotation_residual_deg=float(np.rad2deg(np.linalg.norm(res[3:])/.08)),joint_qpos=q.tolist(),joint_margin_rad=np.minimum(q-m.jnt_range[arm,0],m.jnt_range[arm,1]-q).tolist(),contacts=contact,blockers=bad);report['frames'].append(row)
  if row['position_residual_m']>.007 or row['rotation_residual_deg']>1 or bad:report.update(passed=False,first_failure=row);return report
 report['open_grasp_geometry']=geometry(m,d,zone,site,g._left,g._right);report['passed']=report['open_grasp_geometry']['eligible'];report['closure_and_held_transport_pending']=True
 return report
def main():
 if(HERE/'report.json').exists():raise RuntimeError('one immutable copied candidate; no retries')
 start=time.monotonic();r=dict(instance='R4-04',kind='ONE_WHOLE_SOURCE_PLUS80MM_COPIED_LAYOUT_NOT_PHYSICAL_QUALIFICATION',source_shift_x_m=SHIFT,nominal_xy_m=[.08,-.183955466412554],actual_controls=0,physical_replays=0,GPU_calls=0,models=[],initial=[],error=None,passed=False,old_sources_unchanged=True)
 try:
  for seed in range(10):
   s=Scene(seed);om,od=s.backend.sim.model._model,s.backend.sim.data._data;before=state(om,od);oldhash=digest_model(om)
   try:
    m,d,obj,jid,adr=copy_source(s);row=initial(s,m,d,obj);row['seed']=seed;r['initial'].append(row)
    if not row['passed']:r['stop_reason']='whole initial support/contact/public-frustum failure';break
    if seed==0:route_input=(m,d,obj,jid,adr,s.grasp_assist._center,s.grasp_assist._left,s.grasp_assist._right)
   finally:
    row['original_integration_unchanged']=bool(np.array_equal(before,state(om,od)));row['original_model_unchanged']=oldhash==digest_model(om);s.close()
  if len(r['initial'])==10 and all(x['passed']for x in r['initial']):
   # Constructor only, no backend.step. Same seed0 copied state; no physics
   # qualification replay or altered old trace.
   s=Scene(0);om,od=s.backend.sim.model._model,s.backend.sim.data._data;before=state(om,od);oldhash=digest_model(om)
   try:
    m,d,obj,jid,adr=copy_source(s);r['route']=route(s,m,d,obj,jid,adr);r['passed']=r['route']['passed'];r['stop_reason']=None if r['passed']else'first copied whole-arm approach failure'
   finally:r['original_route_integration_unchanged']=bool(np.array_equal(before,state(om,od)));r['original_route_model_unchanged']=oldhash==digest_model(om);s.close()
 except BaseException as exc:r['error']=f'{type(exc).__name__}:{exc}'
 r['elapsed_s']=time.monotonic()-start;(HERE/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in('error','passed','elapsed_s')},indent=2))
if __name__=='__main__':main()
