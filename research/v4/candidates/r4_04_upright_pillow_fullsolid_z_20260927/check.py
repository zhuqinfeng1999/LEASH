"""One compiled upright rack + continuous copied route, no physics step."""
import copy,inspect,json,time,hashlib
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix,get_camera_extrinsic_matrix
from .scene import Scene,HERE,ROOT,POINT,lower_support
from research.v4.candidates.r4_04_visible_wood_riser_20260927 import check as wood

def initial(s,m,d,obj):
 bid=m.body(obj.root_body).id;points=wood.parent.geom_points(m,d,m.geom('adult_pillow_1_pillow_body_collision').id);rack=m.body('r404_upright_wood_rack').id;R=d.xmat[rack].reshape(3,3);p=d.xpos[rack];local=(points-p)@R;actual_lower=[]
 for gi in range(m.ngeom):
  if(m.geom(gi).name or'').startswith('adult_pillow_1_')and(m.geom_contype[gi]or m.geom_conaffinity[gi]):actual_lower.extend(lower_support(m,d.geom_xpos[gi],d.geom_xmat[gi].reshape(3,3),gi))
 actual_lower=np.array(actual_lower);floorz=actual_lower[:,2].min();bottom=(actual_lower[np.abs(actual_lower[:,2]-floorz)<1e-7]-p)@R;com=(d.xipos[bid]-p)@R
 decklo=np.array([-.325,-.13]);deckhi=np.array([.325,.02]);bottom_supported=bool(np.all(bottom[:,:2]>=decklo)and np.all(bottom[:,:2]<=deckhi));com_supported=bool(np.all(com[:2]>=decklo)and np.all(com[:2]<=deckhi));rackgeoms=[i for i in range(m.ngeom)if(m.geom(i).name or'').startswith('r404_upright_wood_')and m.geom_group[i]==0];allpoints=np.vstack([points]+[wood.parent.geom_points(m,d,i)for i in rackgeoms]);tid=m.geom('table_collision').id;lo=d.geom_xpos[tid,:2]-m.geom_size[tid,:2];hi=d.geom_xpos[tid,:2]+m.geom_size[tid,:2];full=bool(np.all(allpoints[:,:2]>=lo-1e-7)and np.all(allpoints[:,:2]<=hi+1e-7))
 cts=wood.contacts(m,d);bad=[]
 for c in cts:
  if c['distance_m']<0:
   pair=c['pair'];r=any(x.startswith('r404_upright_wood_')for x in pair)
   if c['robot']or c['person']or(c['pillow']and not r):bad.append(c)
 deep=[c for c in cts if c['distance_m']<-.001]
 proxy=SimpleNamespace(model=s.backend.sim.model,data=SimpleNamespace(_data=d,cam_xpos=d.cam_xpos,cam_xmat=d.cam_xmat));zone=d.site_xpos[m.site('v42_adult_pillow_1_grasp_zone').id].copy();views={};groups=np.ones(6,np.uint8);groups[0]=0
 for cam in('agentview','robot0_eye_in_hand'):
  K=get_camera_intrinsic_matrix(proxy,cam,480,640);T=get_camera_extrinsic_matrix(proxy,cam);inv=np.linalg.inv(T)
  def project(v):
   cv=(inv@np.c_[v,np.ones(len(v))].T)[:3];q=K@cv;q=q[:2]/q[2];return dict(bbox_xyxy=[*q.min(1).tolist(),*q.max(1).tolist()],in_frame=bool(np.all(cv[2]>0)and q[0].min()>=0 and q[1].min()>=0 and q[0].max()<640 and q[1].max()<480))
  ray=zone-T[:3,3];ray/=np.linalg.norm(ray);hit=np.array([-1],np.int32);mujoco.mj_ray(m,d,T[:3,3],ray,groups,1,-1,hit);baby=[project(wood.parent.geom_points(m,d,g))for g in range(m.ngeom)if(m.geom(g).name or'').startswith('crib_1_doll_')and m.geom_group[g]==1]
  views[cam]=dict(zone=project(zone[None]),pillow=project(points),infant=baby,seam_ray_hit=m.geom(int(hit[0])).name if hit[0]>=0 else None)
 key=bool(all(v['zone']['in_frame']and(v['seam_ray_hit']or'').startswith('adult_pillow_1_')and all(x['in_frame']for x in v['infant'])for v in views.values())and views['agentview']['pillow']['in_frame'])
 out=dict(actual_static_support=dict(bottom_vertex_count=len(bottom),lowest_surface_above_wood_top_m=float(floorz-.915),lower_vertices_inside_real_deck=bottom_supported,CoM_projection_inside_deck=com_supported,CoM_rack_local=com.tolist()),whole_rack_pillow_table_XY=full,pillow_world_bounds=[points.min(0).tolist(),points.max(0).tolist()],whole_world_bounds=[allpoints.min(0).tolist(),allpoints.max(0).tolist()],contacts=cts,blockers=bad,deep_contacts=deep,views=views,public_key_visibility=key,geometric_support_only_not_dynamics=True)
 out['passed']=bool(bottom_supported and com_supported and full and abs(out['actual_static_support']['lowest_surface_above_wood_top_m'])<1e-6 and not bad and not deep and key);return out

def route_fn():
 src=inspect.getsource(wood.parent.route);old="d.xmat[bid].reshape(3,3)@Rotation.from_euler('z',-np.pi/2).as_matrix()@np.asarray(old['SIDE_R'])";assert src.count(old)==1
 expression="np.stack([d.xmat[bid].reshape(3,3)[:,2],np.cross(np.array([0.,0.,-1.]),d.xmat[bid].reshape(3,3)[:,2]),np.array([0.,0.,-1.])],axis=1)"
 src=src.replace(old,expression).replace(" or row['rotation_residual_deg']>1",'');ns=dict(vars(wood.parent));ns['cts']=wood.contacts;exec(compile(src,__file__,'exec'),ns);return ns['route']

def main():
 out=HERE/'report.json'
 if out.exists():raise RuntimeError('One immutable upright candidate only')
 start=time.monotonic();r=dict(kind='ONE_UPRIGHT_ADULT_PILLOW_ORIGINAL_SEAM_REAL_WOOD_RACK_COPY_ONLY',physical_controls=0,GPU=False,models=[],passed=False,error=None)
 parent=json.loads((ROOT/'research/v4/candidates/r4_04_visible_wood_riser_20260927/source_freeze.json').read_text());parent=parent.get('source_sha256',parent);r['parent_source_drift']=[k for k,v in parent.items()if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
 if r['parent_source_drift']:raise RuntimeError('Parent drift')
 s=Scene(0);om,od=s.backend.sim.model._model,s.backend.sim.data._data;before=wood.parent.state(om,od);mh=wood.parent.digest_model(om)
 try:
  r['scene_initial']=s.upright_initial;m=copy.deepcopy(om);d=mujoco.MjData(m);mujoco.mj_setState(m,d,before,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);obj=s.backend.objects_dict[s.item_name];jid=m.joint(obj.joints[0]).id;adr=m.jnt_qposadr[jid];r['initial']=initial(s,m,d,obj)
  if not r['initial']['passed']:r['stop_reason']='first real compiled upright support/contact/table/public-key failure'
  else:
   r['route']=route_fn()(s,m,d,obj,jid,adr)
   if not r['route']['passed']:r['stop_reason']='first copied whole-arm approach reach/contact/real-region failure'
   else:r['close_and_virtual_held']=wood.make_held()(s,m,d,obj,jid,adr);r['passed']=bool(r['close_and_virtual_held']['passed']);r['stop_reason']=None if r['passed']else'first copied close or held route failure'
 except BaseException as exc:r['error']=f'{type(exc).__name__}: {exc}'
 finally:r['original_integration_unchanged']=bool(np.array_equal(before,wood.parent.state(om,od)));r['original_model_unchanged']=mh==wood.parent.digest_model(om);s.close()
 r['elapsed_s']=time.monotonic()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r.get(k)for k in('passed','error','stop_reason','elapsed_s')}))
if __name__=='__main__':main()
