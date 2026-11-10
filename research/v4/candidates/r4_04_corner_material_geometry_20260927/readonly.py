"""One copied-data actual-material whole-approach witness, no physical steps."""
import copy,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation,Slerp
from research.v4.accelerated.tabletop import PARK_QPOS
from research.v4.candidates.r4_03_neck_layout_readonly_20260927.check_geometry import geometry,state
from .scene import Scene,HERE,ROOT,CORNER
def cts(m,d):
 rows=[]
 for c in d.contact[:d.ncon]:
  names=[m.geom(int(g)).name or''for g in(c.geom1,c.geom2)];robot=any(n.startswith(('robot0_','gripper0_'))for n in names);pillow=any(n.startswith('adult_pillow_1_')for n in names)
  if robot or pillow:rows.append(dict(pair=names,distance_m=float(c.dist),world_pos_m=c.pos.tolist(),robot=robot,pillow=pillow,table='table_collision'in names,infant=any(n.startswith('crib_1_doll_')for n in names),palm_or_arm=any(n.startswith('robot0_')or n=='gripper0_right_hand_collision'for n in names)))
 return rows
def main():
 if (HERE/'report.json').exists():raise RuntimeError('one fresh readonly candidate only')
 s=Scene(0);m=s.backend.sim.model._model;d=s.backend.sim.data._data;before=state(m,d);model_hash=hashlib.sha256(m.geom_size.tobytes()+m.mesh_vert.tobytes()+m.site_pos.tobytes()).hexdigest();shadow=mujoco.MjData(m);mujoco.mj_setState(m,shadow,before,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,shadow)
 report=dict(kind='COPIED_MJDATA_IK_MATERIAL_WITNESS_NOT_PHYSICAL_ACCEPTANCE',seed=0,actual_controls=0,GPU_calls=0,model_calls=[],error=None,candidate_initial=s.candidate_initial)
 try:
  obj=s.backend.objects_dict[s.item_name];gid=m.geom('adult_pillow_1_pillow_body_collision').id;mid=m.geom_dataid[gid];start=m.mesh_vertadr[mid];num=m.mesh_vertnum[mid];v=m.mesh_vert[start:start+num];bid=m.body(obj.root_body).id
  # MuJoCo mesh compiler rebases vertices; transform compiled vertices back
  # into their actual author/body frame, not assume OBJ coordinates survive.
  vb=(d.xmat[bid].reshape(3,3).T@(d.geom_xpos[gid]+v@d.geom_xmat[gid].reshape(3,3).T-d.xpos[bid]).T).T
  hull=ConvexHull(vb);p=np.array(CORNER);lower=[];upper=[]
  for a,b,c,k in hull.equations:
   if c>1e-7:upper.append(-(a*p[0]+b*p[1]+k)/c)
   elif c< -1e-7:lower.append(-(a*p[0]+b*p[1]+k)/c)
  lo=max(lower);hi=min(upper);margin=-(hull.equations[:,:3]@p+hull.equations[:,3]);idx=int(np.argmin(margin))
  report['actual_compiled_material']=dict(body_frame_bounds_m=[vb.min(0).tolist(),vb.max(0).tolist()],corner_m=p.tolist(),corner_material_z_m=[lo,hi],actual_local_thickness_m=hi-lo,inside_actual_collision_visual_material=bool(np.all(margin>=-1e-7)),closest_surface_normal_body=hull.equations[idx,:3].tolist(),closest_normal_clearance_m=float(margin[idx]),same_mesh_visual_collision=bool(m.geom_dataid[m.geom('adult_pillow_1_pillow_body_visual').id]==mid))
  assist=s.grasp_assist;site=assist._center;zone=m.site('v42_adult_pillow_1_grasp_zone').id;arm=[m.joint(f'robot0_joint{i}').id for i in range(1,8)];adrs=[m.jnt_qposadr[i]for i in arm];q=np.asarray(shadow.qpos[adrs]).copy();ref=mujoco.MjData(m);mujoco.mj_setState(m,ref,before,mujoco.mjtState.mjSTATE_INTEGRATION)
  ref.qpos[adrs]=PARK_QPOS;mujoco.mj_forward(m,ref);target_R=ref.site_xmat[site].reshape(3,3).copy();initial_R=shadow.site_xmat[site].reshape(3,3).copy();initial_xyz=shadow.site_xpos[site].copy();zone_xyz=shadow.site_xpos[zone].copy();above=np.array([zone_xyz[0],zone_xyz[1],1.2]);waypoints=[]
  interp=Slerp([0,1],Rotation.from_matrix([initial_R,target_R]))
  for t in np.linspace(0,1,9)[1:]:waypoints.append(('orientation_at_reset',initial_xyz,interp([t]).as_matrix()[0]))
  for t in np.linspace(0,1,13)[1:]:waypoints.append(('above_actual_corner',initial_xyz+t*(above-initial_xyz),target_R))
  for t in np.linspace(0,1,17)[1:]:waypoints.append(('descend_to_actual_corner',above+t*(zone_xyz-above),target_R))
  frames=[]
  def error(qp,xyz,R):
   shadow.qpos[adrs]=qp;mujoco.mj_forward(m,shadow);return np.r_[shadow.site_xpos[site]-xyz,.08*Rotation.from_matrix(R@shadow.site_xmat[site].reshape(3,3).T).as_rotvec()]
  for number,(phase,xyz,R)in enumerate(waypoints):
   opt=least_squares(error,q,args=(xyz,R),bounds=(m.jnt_range[arm,0]+1e-6,m.jnt_range[arm,1]-1e-6),max_nfev=160,xtol=1e-10,ftol=1e-10,gtol=1e-10);q=opt.x;res=error(q,xyz,R);contact=cts(m,shadow)
   row=dict(sequence=number,phase=phase,position_error_m=float(np.linalg.norm(res[:3])),rotation_error_rad=float(np.linalg.norm(res[3:])/.08),copied_arm_qpos=q.tolist(),eef_xyz_m=shadow.site_xpos[site].tolist(),contacts=contact);frames.append(row)
   if row['position_error_m']>.007 or row['rotation_error_rad']>np.deg2rad(1):report['first_reach_failure']=row;break
  g=geometry(m,shadow,zone,site,assist._left,assist._right);blocking=[dict(frame=f['sequence'],phase=f['phase'],contact=c)for f in frames for c in f['contacts']if c['robot']and c['distance_m']<-.001 and(c['table']or c['pillow']or c['infant'])]
  report.update(whole_copied_approach=frames,open_final_grasp_geometry=g,open_finger_qpos={m.joint(i).name:float(shadow.qpos[m.jnt_qposadr[i]])for i in range(m.njnt)if'finger_joint'in(m.joint(i).name or'')},copied_open_approach_blocking=blocking,original_initial_contacts=cts(m,d),geometry_eligible_and_clean=bool(report['actual_compiled_material']['inside_actual_collision_visual_material']and len(frames)==len(waypoints)and g['eligible']and not blocking),actual_reach_grasp_dynamics_unproven=True,limitations=['Copied IK is an instantaneous geometry hypothesis, not real controller reach/trajectory proof.','Pillow18.869mm material is thinner than open pads; copied closing overlap is not dynamic impossibility. No pixel/native acceptance from this CPU test.'])
 except BaseException as e:report['error']=f'{type(e).__name__}:{e}';report['geometry_eligible_and_clean']=False
 finally:
  report['original_integration_unchanged']=bool(np.array_equal(before,state(m,d)));report['original_model_mesh_geometry_unchanged']=model_hash==hashlib.sha256(m.geom_size.tobytes()+m.mesh_vert.tobytes()+m.site_pos.tobytes()).hexdigest();s.close();(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k]for k in('error','geometry_eligible_and_clean','original_integration_unchanged','original_model_mesh_geometry_unchanged')}))
if __name__=='__main__':main()
