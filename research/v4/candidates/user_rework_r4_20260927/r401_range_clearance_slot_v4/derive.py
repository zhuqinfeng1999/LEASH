"""One actual legacy model -> continuous whole-solid slot bound; no controls."""
import hashlib,itertools,json,math,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..r401_normal_crib_legacy_slot_v2 import scene as old
HERE=Path(__file__).resolve().parent;ROOT=old.ROOT
def corners(m,d,g):
 a=m.geom_aabb[g];v=np.asarray([a[:3]+a[3:]*np.array(c)for c in itertools.product((-1,1),repeat=3)])
 return d.geom_xpos[g]+v@d.geom_xmat[g].reshape(3,3).T
def minimum_y(p,angle):
 x,y=float(p[0]),float(p[1]);choices=[-angle,angle]
 stationary=math.atan2(x,y)
 for k in range(-2,3):
  t=stationary+k*math.pi
  if-angle<=t<=angle:choices.append(t)
 return min((x*math.sin(t)+y*math.cos(t),t)for t in choices)
def main():
 out=HERE/'geometry_range.json'
 if out.exists():raise RuntimeError('One analytic derivation only')
 start=time.monotonic();s=old.Scene(1)
 try:
  sim=s.env.sim;m,d=sim.model._model,sim.data._data;tid=m.geom('table_collision').id;tv=corners(m,d,tid);obj=s.backend.objects_dict[s.target_name];q=sim.data.get_joint_qpos(obj.joints[0]).copy();R=Rotation.from_quat(q[[4,5,6,3]]).as_matrix();rows=[];all_points=[]
  for g in range(m.ngeom):
   n=m.geom(g).name or''
   if not n.startswith('crib_1_')or not(m.geom_contype[g]or m.geom_conaffinity[g]):continue
   points=(corners(m,d,g)-q[:3])@R;all_points.extend(points.tolist());extrema=[minimum_y(p,math.radians(10))for p in points];minimum,theta=min(extrema)
   rows.append(dict(geom=n,type=int(m.geom_type[g]),body_local_aabb_corners=points.tolist(),continuous_yaw_min_local_y_m=minimum,yaw_at_min_deg=math.degrees(theta)))
  minimum=min(r['continuous_yaw_min_local_y_m']for r in rows);table_ymax=float(tv[:,1].max());need=table_ymax+.020+.005-minimum;nominal=math.ceil(need*1e6)/1e6
  contacts=[dict(geoms=[m.geom(int(i)).name or''for i in(c.geom1,c.geom2)],distance_m=float(c.dist))for c in d.contact[:d.ncon]]
  record=dict(kind='EXACT_REAL_LEGACY_TABLE_CONTINUOUS_FULL_SOLID_BOUND',actual_old_seed=1,actual_controls=0,sim_time_s=float(d.time),table=dict(name='table_collision',geom_xpos=d.geom_xpos[tid].tolist(),geom_xmat=d.geom_xmat[tid].reshape(3,3).tolist(),size=m.geom_size[tid].tolist(),world_bbox=[tv.min(0).tolist(),tv.max(0).tolist()]),crib=dict(xml=str(old.furniture.XML.relative_to(ROOT)),actual_seed1_qpos=q.tolist(),solid_geoms=len(rows),full_conservative_body_local_bbox=[np.min(all_points,0).tolist(),np.max(all_points,0).tolist()],rows=rows),domain=dict(center_perturbation_xy_m=[-.020,.020],yaw_deg=[-10,10],continuous_extrema_not_grid=True,all_collision_solid_geometry_included=True),derivation=dict(required_whole_crib_table_y_gap_m=.005,minimum_rotated_local_y_m=minimum,old_nominal_crib_y_m=.90,minimum_required_nominal_y_m=need,new_nominal_crib_y_m=nominal,uniform_translation_y_m=nominal-.90,round_up_margin_m=nominal-need,proved_all_solid_minimum_y_m=nominal-.020+minimum,table_max_y_m=table_ymax,minimum_whole_crib_table_y_gap_m=nominal-.020+minimum-table_ymax,conservative_geom_local_AABB_union=True),old_actual_contacts=contacts,old_source_sha256=json.loads((old.HERE/'source_freeze.json').read_text()),wall_s=time.monotonic()-start,GPU_models_called=[])
  out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record['derivation']))
 finally:s.close()
if __name__=='__main__':main()
