"""Single world-mesh PCA plane close view; no angle sweep, controls or render."""
import json,hashlib,math
from pathlib import Path
import numpy as np,mujoco
from research.v4.accelerated.tabletop import create_scene
from research.v4.candidates.panda_glyph_hide_visual_20260927 import helper
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]

def world_mesh(m,d,g):
 mid=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);return np.asarray(m.mesh_vert[a:a+n],float)@np.asarray(d.geom_xmat[g]).reshape(3,3).T+np.asarray(d.geom_xpos[g])
def main():
 s=create_scene('R2-01',0,camera_observations=False)
 try:
  m,d=s.sim.model._model,s.sim.data._data;g=helper.identify(m);v=world_mesh(m,d,g);c=v.mean(axis=0);evals,evecs=np.linalg.eigh(np.cov((v-c).T));normal=evecs[:,0];bid=int(m.geom_bodyid[g]);bodycenter=np.asarray(d.xpos[bid]);dot=float(normal@(c-bodycenter))
  if abs(dot)<.002:raise RuntimeError('Outward glyph plane sign ambiguous; stop rather than sweep')
  if dot<0:normal=-normal
  # Camera looks along inward surface normal. Original shell stays visible.
  z=normal;right=np.cross([0.,0.,1.],z)
  if np.linalg.norm(right)<.1:raise RuntimeError('Camera vertical-plane degeneracy; no angle alternative')
  right/=np.linalg.norm(right);up=np.cross(z,right);q=v-c;x=q@right;y=q@up;out=q@normal;aspect=1920/1080;fovy=40.;tan=math.tan(math.radians(fovy/2));distance=max(float(np.max(out+np.maximum(np.abs(x)/(tan*aspect*.60),np.abs(y)/(tan*.60))))+.025,.12)
  pos=c+distance*normal;delta=pos-c
  pose=dict(look_at_world_m=c.tolist(),position_world_m=pos.tolist(),distance_m=distance,azimuth_deg=(math.degrees(math.atan2(delta[1],delta[0]))+180)%360,elevation_deg=-math.degrees(math.asin(delta[2]/np.linalg.norm(delta))),fovy_deg=fovy,online=False,derived_from='actual worldglyph vertex PCA least-variance normal, outward sign versus actual link5 body origin',normal_world=normal.tolist())
  depth=distance-out;ndc=np.c_[x/(depth*tan*aspect),y/(depth*tan)]
  # Sample real vertex sight lines in the unchanged model; no hand-authored mask.
  idx=np.linspace(0,len(v)-1,min(40,len(v)),dtype=int);occ=[]
  groups=np.array([0,1,0,0,0,0],dtype=np.uint8)
  for i in idx:
   ray=v[i]-pos;length=np.linalg.norm(ray);hit=np.array([-1],dtype=np.int32);r=mujoco.mj_ray(m,d,pos,ray/length,groups,1,-1,hit)
   occ.append(dict(vertex_index=int(i),first_geom_id=int(hit[0]),first_geom_name=m.geom(int(hit[0])).name if hit[0]>=0 else None,hit_distance_m=float(r),vertex_distance_m=float(length),occluded_before_vertex=bool(r>=0 and r<length-.001 and hit[0]!=g)))
  fields=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos');raw=s._get_observations(force_update=True);oldpath=ROOT/'research/v4/candidates/panda_glyph_native_pair_20260927/native_pair/report.json';old=json.loads(oldpath.read_text());proprio={k:dict(equal=bool(np.array_equal(np.asarray(raw[k]),np.asarray(old['fresh_baseline_proprio'][k]['value']))),actual=np.asarray(raw[k]).tolist())for k in fields}
  if not all(r['equal']for r in proprio.values()):raise RuntimeError('Current initialized robot state differs from actual prior frozen native packet')
  r=dict(passed=True,pose=pose,glyph_geom_id=g,glyph_geom_name=m.geom(g).name,glyph_mesh_signature=helper.mesh_signature(m,int(m.geom_dataid[g])),actual_world_centroid_m=c.tolist(),actual_world_bounds_m=[v.min(axis=0).tolist(),v.max(axis=0).tolist()],pca_eigenvalues_m2=evals.tolist(),outward_sign_reference_body_origin_m=bodycenter.tolist(),outward_signed_distance_m=abs(dot),glyph_NDC_bounds=[ndc.min(axis=0).tolist(),ndc.max(axis=0).tolist()],occlusion_probe=dict(sampled_vertices=len(occ),occluded_before_vertex=sum(o['occluded_before_vertex']for o in occ),samples=occ),same_prior_native_proprio=proprio,actual_qpos=np.asarray(d.qpos).tolist(),actual_qvel=np.asarray(d.qvel).tolist(),simulation_time_s=float(d.time),controls=0,render_calls=0,GPU=False,pose_grid=False,source_prior_report_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),native_appearance_acceptance='pending_root_actual_close_view')
  (HERE/'camera_fit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,pose=pose,ndc=r['glyph_NDC_bounds'],occluded_samples=r['occlusion_probe']['occluded_before_vertex'])))
 finally:s.close()
if __name__=='__main__':main()
