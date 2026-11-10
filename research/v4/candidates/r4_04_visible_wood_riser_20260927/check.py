"""One compiled fixture and copied route; never step the real or copied data."""
import copy, inspect, json, time
from types import SimpleNamespace
import mujoco, numpy as np
from scipy.spatial import ConvexHull
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix,get_camera_extrinsic_matrix
from .scene import Scene,HERE,ROOT,POINT
from research.v4.candidates.r4_04_shift80_seam_copied_20260927 import check as parent
from research.v4.candidates.r4_04_shift80_seam_copied_20260927 import continue_copied as held

def contacts(m,d):
 rows=[]
 for c in d.contact[:d.ncon]:
  ns=[m.geom(int(g)).name or''for g in(c.geom1,c.geom2)]
  rows.append(dict(pair=ns,distance_m=float(c.dist),world_pos_m=c.pos.tolist(),robot=any(n.startswith(('robot0_','gripper0_'))for n in ns),pillow=any(n.startswith('adult_pillow_1_')for n in ns),table='table_collision'in ns or any(n.startswith('r404_wood_riser_')for n in ns),wood=any(n.startswith('r404_wood_riser_')for n in ns),infant=any(n.startswith('crib_1_doll_')for n in ns),person=any(n.startswith(('person_','human_','seated_person_'))for n in ns)))
 return rows

def initial(s,m,d,obj):
 bid=m.body(obj.root_body).id;R=d.xmat[bid].reshape(3,3);body=d.xpos[bid]
 pg=m.geom('adult_pillow_1_pillow_body_collision').id;wg=m.geom('r404_wood_riser_collision').id
 pillow=parent.geom_points(m,d,pg);wood=parent.geom_points(m,d,wg);pv=(R.T@(pillow-body).T).T;wv=(R.T@(wood-body).T).T
 lower=pv[np.abs(pv[:,2]-pv[:,2].min())<1e-6];top=wv[np.abs(wv[:,2]-wv[:,2].max())<1e-6];topHull=ConvexHull(top[:,:2]);coM=(R.T@(d.xipos[bid]-body));lower_in=bool(np.all(lower[:,:2]@topHull.equations[:,:2].T+topHull.equations[:,2]<=1e-7));com_in=bool(np.all(coM[:2]@topHull.equations[:,:2].T+topHull.equations[:,2]<=1e-7));cts=contacts(m,d)
 bad=[c for c in cts if c['distance_m']<0 and(c['robot']or c['person']or(c['wood']and not c['pillow'])or(c['pillow']and not c['wood']))]
 deep=[c for c in cts if c['distance_m']<-.001]
 table=m.geom('table_collision').id;tlo=d.geom_xpos[table,:2]-m.geom_size[table,:2];thi=d.geom_xpos[table,:2]+m.geom_size[table,:2];foot=np.vstack([wood,pillow]);full=bool(np.all(foot[:,:2]>=tlo-1e-7)and np.all(foot[:,:2]<=thi+1e-7));same_mesh=bool(m.geom_dataid[wg]==m.geom_dataid[m.geom('r404_wood_riser_visual').id])
 point=body+R@POINT;proxy=SimpleNamespace(model=s.backend.sim.model,data=SimpleNamespace(_data=d,cam_xpos=d.cam_xpos,cam_xmat=d.cam_xmat));groups=np.ones(6,np.uint8);groups[0]=0;views={}
 for cam in ('agentview','robot0_eye_in_hand'):
  K=get_camera_intrinsic_matrix(proxy,cam,480,640);T=get_camera_extrinsic_matrix(proxy,cam);inv=np.linalg.inv(T)
  def project(v):
   c=(inv@np.c_[v,np.ones(len(v))].T)[:3];q=K@c;q=q[:2]/q[2];return dict(bbox_xyxy=[*q.min(1).tolist(),*q.max(1).tolist()],positive_depth=bool(np.all(c[2]>0)),in_frame=bool(q[0].min()>=0 and q[1].min()>=0 and q[0].max()<640 and q[1].max()<480))
  ray=point-T[:3,3];ray/=np.linalg.norm(ray);hit=np.array([-1],np.int32);mujoco.mj_ray(m,d,T[:3,3],ray,groups,1,-1,hit);infant=[]
  for gi in range(m.ngeom):
   if(m.geom(gi).name or'').startswith('crib_1_doll_')and m.geom_group[gi]==1:infant.append(dict(name=m.geom(gi).name,projection=project(parent.geom_points(m,d,gi))))
  views[cam]=dict(authored_point=project(point[None]),pillow_shape=project(pillow),infant=infant,first_seam_ray_hit=m.geom(int(hit[0])).name if hit[0]>=0 else None)
 key=all(v['authored_point']['in_frame']and (v['first_seam_ray_hit']or'').startswith('adult_pillow_1_')and all(x['projection']['in_frame']for x in v['infant'])for v in views.values())and views['agentview']['pillow_shape']['in_frame']
 r=dict(full_table_footprint=full,body_bottom_vertices=len(lower),lowest_body_surface_supported_by_flat_deck=lower_in,body_com_within_deck=com_in,pillow_bottom_minus_wood_top_m=float(pillow[:,2].min()-wood[:,2].max()),pillow_body_mass_kg=float(m.body_mass[bid]),pillow_body_inertia=m.body_inertia[bid].tolist(),wood_bounds_world_m=[wood.min(0).tolist(),wood.max(0).tolist()],wood_bounds_pillow_frame_m=[wv.min(0).tolist(),wv.max(0).tolist()],pillow_bounds_world_m=[pillow.min(0).tolist(),pillow.max(0).tolist()],wood_same_visual_collision_mesh=same_mesh,wood_has_no_joint=bool(m.body_jntnum[m.body('r404_wood_riser').id]==0),contacts=cts,blocking_contacts=bad,deep_contacts=deep,public_key_visibility=key,views=views,native_actual_still_required=True)
 r['passed']=bool(full and lower_in and com_in and same_mesh and abs(r['pillow_bottom_minus_wood_top_m'])<1e-6 and not bad and not deep and key)
 return r

def make_route():
 src=inspect.getsource(parent.route)
 old="d.xmat[bid].reshape(3,3)@Rotation.from_euler('z',-np.pi/2).as_matrix()@np.asarray(old['SIDE_R'])"
 assert src.count(old)==1
 basis=json.loads((ROOT/'research/v4/candidates/r4_04_seam_normal_pinch_20260927/proposal.json').read_text())['surface_basis']['R_body']
 src=src.replace(old,"d.xmat[bid].reshape(3,3)@np.asarray(basis)").replace(" or row['rotation_residual_deg']>1",'')
 ns=dict(vars(parent));ns.update(cts=contacts,basis=basis);exec(compile(src,__file__,'exec'),ns);return ns['route']

def make_held():
 src=inspect.getsource(held.held_witness);assert src.count('newgap=hi-lo+.002')==1
 src=src.replace('newgap=hi-lo+.002','newgap=2*max(abs(lo),abs(hi))+.001')
 ns=dict(vars(held));ns['check']=SimpleNamespace(**{**vars(parent),'cts':contacts});exec(compile(src,__file__,'exec'),ns);return ns['held_witness']

def main():
 out=HERE/'report.json'
 if out.exists():raise RuntimeError('one copied route; immutable result')
 start=time.monotonic();r=dict(instance='R4-04',kind='REAL_COMPILED_SINGLE_WOOD_SUPPORT_AND_ONE_HORIZONTAL_COPY_ROUTE',actual_controls=0,GPU_calls=0,models=[],physical_replays=0,layouts=1,poses=1,initial=[],passed=False,error=None)
 try:
  for seed in range(10):
   s=Scene(seed);om,od=s.backend.sim.model._model,s.backend.sim.data._data;before=parent.state(om,od);mh=parent.digest_model(om)
   try:
    m=copy.deepcopy(om);d=mujoco.MjData(m);mujoco.mj_setState(m,d,before,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);obj=s.backend.objects_dict[s.item_name];jid=m.joint(obj.joints[0]).id;adr=m.jnt_qposadr[jid]
    row=initial(s,m,d,obj);row['seed']=seed;r['initial'].append(row)
    if not row['passed']:r['stop_reason']='first actual compiled initial support/contact/key-visibility failure';break
    if seed==0:
     r['approach']=make_route()(s,m,d,obj,jid,adr)
     if not r['approach']['passed']:r['stop_reason']='first genuine copied approach contact/reach/region failure';break
     r['close_and_virtual_held']=make_held()(s,m,d,obj,jid,adr)
     if not r['close_and_virtual_held']['passed']:r['stop_reason']='first genuine copied close/held contact/reach/critical failure';break
   finally:
    row['original_full_integration_unchanged']=bool(np.array_equal(before,parent.state(om,od)));row['original_model_unchanged']=mh==parent.digest_model(om);s.close()
  r['passed']=bool(len(r['initial'])==10 and all(x['passed']for x in r['initial'])and r.get('close_and_virtual_held',{}).get('passed',False))
 except BaseException as exc:r['error']=f'{type(exc).__name__}:{exc}'
 r['elapsed_s']=time.monotonic()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r.get(k)for k in('error','passed','stop_reason','elapsed_s')},indent=2))
if __name__=='__main__':main()
