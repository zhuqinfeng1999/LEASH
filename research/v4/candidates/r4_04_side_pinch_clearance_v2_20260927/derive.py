"""Analytic minimal normal wrist pitch and outside-cloth prepoint, no controls."""
import itertools,json,math
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.r4_04_analytical_supported_layout_20260927.scene import Scene,SIDE_R
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def geom_points(m,d,g):
 kind=int(m.geom_type[g])
 if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
  mid=m.geom_dataid[g];a=m.mesh_vertadr[mid];n=m.mesh_vertnum[mid];v=m.mesh_vert[a:a+n]
 else:
  bound=m.geom_aabb[g];v=np.asarray([bound[:3]+bound[3:]*x for x in itertools.product((-1,1),repeat=3)])
 return d.geom_xpos[g]+v@d.geom_xmat[g].reshape(3,3).T
def main():
 if(HERE/'proposal.json').exists():raise RuntimeError('one analytic pitch/prepoint only')
 old=ROOT/'research/v4/candidates/r4_04_analytical_supported_layout_20260927/report.json';record=json.loads(old.read_text());s=Scene(0);m,d=s.backend.sim.model._model,s.backend.sim.data._data;shadow=mujoco.MjData(m);shadow.qpos[:]=d.qpos
 try:
  arm=[m.jnt_qposadr[m.joint(f'robot0_joint{i}').id]for i in range(1,8)];shadow.qpos[arm]=record['whole_copied_approach'][-1]['copied_arm_qpos']
  for i in range(m.njnt):
   name=m.joint(i).name or''
   if'finger_joint'in name:shadow.qpos[m.jnt_qposadr[i]]=m.jnt_range[i,1]if name.endswith('joint1')else m.jnt_range[i,0]
  mujoco.mj_forward(m,shadow);sid=s.grasp_assist._center;center=shadow.site_xpos[sid].copy();R=shadow.site_xmat[sid].reshape(3,3);rows=[]
  for g in range(m.ngeom):
   if(m.geom(g).name or'').startswith('gripper0_')and m.geom_group[g]==0:
    p=(R.T@(geom_points(m,shadow,g)-center).T).T;rows.extend((m.geom(g).name,x)for x in p)
  table=m.geom('table_collision').id;top=float(d.geom_xpos[table,2]+m.geom_size[table,2]);target_z=float(d.site_xpos[m.site('v42_adult_pillow_1_grasp_zone').id,2]);tau=top+.003-target_z;feasible=[(0.,math.radians(60))]
  for _,p in rows:
   A=float(p[0]);B=float(-p[2]);r=math.hypot(A,B)
   if r<=abs(tau):continue
   phi=math.atan2(B,A);half=math.acos(tau/r);allowed=[(phi-half+2*k*math.pi,phi+half+2*k*math.pi)for k in range(-2,3)];feasible=[(max(a,c),min(b,e))for a,b in feasible for c,e in allowed if max(a,c)<=min(b,e)]
   if not feasible:raise RuntimeError('no normal<=60degree side pitch clears real gripper/table envelope')
  minimum=min(a for a,b in feasible);chosen=minimum+math.radians(.5)
  if not any(a<=chosen<=b for a,b in feasible):raise RuntimeError('no fixed .5degree numeric margin in feasible pitch interval')
  newR=np.asarray(SIDE_R)@Rotation.from_euler('y',-chosen).as_matrix();rel=np.asarray([p for _,p in rows])@newR.T;body=s.backend.objects_dict[s.item_name];source=np.vstack([geom_points(m,d,m.geom(n).id)for n in body.contact_geoms]);min_y=float(source[:,1].min());zone=d.site_xpos[m.site('v42_adult_pillow_1_grasp_zone').id].copy();pre_y=min_y-float(rel[:,1].max())-.005;dist=float((zone[1]-pre_y)/newR[1,2]);pre=zone-newR[:,2]*dist
  proposal=dict(kind='ONE_ANALYTIC_REAL_GRIPPER_ENVELOPE_SIDE_PITCH_NOT_DYNAMIC',minimal_pitch_deg=math.degrees(minimum),chosen_pitch_deg=math.degrees(chosen),fixed_numeric_pitch_margin_deg=.5,feasible_pitch_intervals_deg=[[math.degrees(a),math.degrees(b)]for a,b in feasible],SIDE_R=newR.tolist(),actual_gripper_points=len(rows),source_frontmost_y_m=min_y,planned_fullopen_gripper_furthest_y_m=float(rel[:,1].max()),source_prepoint_clearance_m=.005,along_finger_front_approach_distance_m=dist,seed0_actual_corner_world_m=zone.tolist(),seed0_derived_prepoint_world_m=pre.tolist(),prepoint_rule='For each actual seed use unchanged-source frontmost collision envelope minus actual authored gripper forward extent and5mm; then approach same real corner along finger axis.',min_gripper_table_clearance_m=float(target_z+rel[:,2].min()-top),table_top_m=top,authored_corner_material_unchanged=True,layout_parent='r4_04_analytical_supported_layout_20260927',actual_controls=0,GPU_calls=0,models=[])
  (HERE/'proposal.json').write_text(json.dumps(proposal,indent=2)+'\n');print(json.dumps(proposal,indent=2))
 finally:s.close()
if __name__=='__main__':main()
