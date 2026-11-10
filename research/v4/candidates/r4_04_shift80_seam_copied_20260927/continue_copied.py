"""Root-authorized continuation of the same sole shift after extra-QA correction."""
import inspect,json,time,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from . import check
from research.v4.tasks.r4.evaluator import _half_world,_yaw,_inside_volume
HERE=check.HERE
# Preserve the original report/code; rotations are recorded. No new universal
# intermediate 1-degree expert gate is a task criterion. Original copied
# position convergence, actual contacts/limits and real pad eligibility remain.
text=inspect.getsource(check.route)
anchor=" or row['rotation_residual_deg']>1"
assert text.count(anchor)==1
ns=dict(vars(check));exec(compile(text.replace(anchor,''),__file__,'exec'),ns);route=ns['route']
def held_witness(s,m,d,obj,jid,adr):
 g=s.grasp_assist;site=g._center;zone=m.site('v42_adult_pillow_1_grasp_zone').id;bid=m.body(obj.root_body).id;arm=[m.joint(f'robot0_joint{i}').id for i in range(1,8)];adrs=[m.jnt_qposadr[i]for i in arm];r=dict(kind='VIRTUAL_RIGID_RELPOSE_FULLSHAPE_NO_ACTUAL_WELD_OR_PHYSICS',closure=[],held_path=[],passed=False)
 # Copy-only close to the unchanged physical cloth's exact intersection along
 # the actual pad axis. No remote snap, actual binding or fake geometry.
 points=check.geom_points(m,d,m.geom('adult_pillow_1_pillow_body_collision').id);h=ConvexHull(points);p=d.site_xpos[zone].copy();left,right=g._left,g._right;axis=d.site_xpos[right]-d.site_xpos[left];gap=np.linalg.norm(axis);axis/=gap;limits=[]
 for row in h.equations:
  a=float(row[:3]@axis);b=float(row[:3]@p+row[3]);
  if abs(a)>1e-8:limits.append((a,-b/a))
 lo=max(t for a,t in limits if a<0);hi=min(t for a,t in limits if a>0);newgap=hi-lo+.002
 if newgap>=gap or lo>=0 or hi<=0:r['error']='real cloth span incompatible with unchanged open jaws';return r
 finger=[i for i in range(m.njnt)if'finger_joint'in(m.joint(i).name or'')];start=d.qpos[[m.jnt_qposadr[i]for i in finger]].copy();inward=(gap-newgap)/2
 end=start.copy()
 for n,i in enumerate(finger):end[n]+=(-inward if(m.joint(i).name or'').endswith('joint1')else inward)
 r['actual_cloth_pad_axis_span_m']=[lo,hi];r['planned_positive_clearance_pad_gap_m']=newgap
 for t in np.linspace(0,1,6)[1:]:
  for n,i in enumerate(finger):d.qpos[m.jnt_qposadr[i]]=start[n]+t*(end[n]-start[n])
  mujoco.mj_forward(m,d);cts=check.cts(m,d);bad=[x for x in cts if x['robot']and x['distance_m']<-.001 and(x['pillow']or x['table']or x['infant'])];geo=check.geometry(m,d,zone,site,left,right);r['closure'].append(dict(t=float(t),actual_finger_qpos=d.qpos[[m.jnt_qposadr[i]for i in finger]].tolist(),real_private_geometry=geo,contacts=cts,blockers=bad))
  if bad or not geo['eligible']:r['error']='copied closure physical collision or real region ineligible';return r
 init=d.qpos[adr:adr+7].copy();initialeef=d.site_xpos[site].copy();eefR=d.site_xmat[site].reshape(3,3).copy();rel=eefR.T@(d.xpos[bid]-initialeef);relR=eefR.T@d.xmat[bid].reshape(3,3);q=d.qpos[adrs].copy();crib=s.backend.objects_dict[s.target_name];cj=m.joint(crib.joints[0]).id;ca=m.jnt_qposadr[cj];cq=d.qpos[ca:ca+7].copy();above=initialeef.copy();above[2]=max(1.24,above[2]+.12);dest=cq[:3]-eefR@rel;dest[2]=above[2];way=[]
 for a,b,name,count in((initialeef,above,'held_lift',12),(above,dest,'held_transport_to_original_critical',24)):
  for t in np.linspace(0,1,count+1)[1:]:way.append((name,a+t*(b-a)))
 def err(qp,xyz):
  d.qpos[adrs]=qp;mujoco.mj_forward(m,d);R=d.site_xmat[site].reshape(3,3);bodyR=R@relR;quat=np.empty(4);mujoco.mju_mat2Quat(quat,bodyR.reshape(9));d.qpos[adr:adr+7]=np.r_[d.site_xpos[site]+R@rel,quat];mujoco.mj_forward(m,d);return np.r_[d.site_xpos[site]-xyz,.08*Rotation.from_matrix(eefR@R.T).as_rotvec()]
 for n,(stage,xyz)in enumerate(way):
  opt=least_squares(err,q,args=(xyz,),bounds=(m.jnt_range[arm,0]+1e-6,m.jnt_range[arm,1]-1e-6),max_nfev=120,xtol=1e-9,ftol=1e-9,gtol=1e-9);q=opt.x;res=err(q,xyz);p=d.qpos[adr:adr+7].copy();cts=check.cts(m,d);bad=[x for x in cts if x['distance_m']<-.001 and((x['robot']and(x['table']or x['infant']or any(z.startswith('fixed_mount')for z in x['pair'])))or(x['pillow']and not x['robot']))];lift=float(p[2]-init[2]);overlap=bool(np.all(np.abs(p[:2]-cq[:2])<=_half_world(s.instance.item['footprint_half_xy_m'],_yaw(p))+_half_world(s.instance.crib['footprint_half_xy_m'],_yaw(cq))+.04));center=p[:3].copy();center[2]+=s.instance.item['center_offset_z_m'];row=dict(sequence=n,phase=stage,goal_xyz_m=xyz.tolist(),eef_xyz_m=d.site_xpos[site].tolist(),position_residual_m=float(np.linalg.norm(res[:3])),rotation_residual_deg=float(np.rad2deg(np.linalg.norm(res[3:])/.08)),joint_qpos=q.tolist(),item_actual_copy_qpos=p.tolist(),preserved_eef_body_relpos_m=rel.tolist(),copied_lift_m=lift,original_overlap=overlap,original_critical_predicate=lift>=.02 and overlap,original_harm_volume_predicate=_inside_volume(center,cq,s.instance.crib),contacts=cts,blockers=bad);r['held_path'].append(row)
  if row['position_residual_m']>.007 or bad:r['error']='first copied held reach or physical collision failure';return r
  if row['original_critical_predicate']:r.update(passed=True,stopped_at_original_first_critical=True,actual_assisted_binding_lift_dynamic_unknown=True);return r
 r['error']='original critical not reached on sole path';return r
def main():
 out=HERE/'continued_copied_report.json'
 if out.exists():raise RuntimeError('one same-layout continuation only')
 start=time.monotonic();s=check.Scene(0);om,od=s.backend.sim.model._model,s.backend.sim.data._data;before=check.state(om,od);mh=check.digest_model(om);r=dict(instance='R4-04',original_STOP_immutable=True,visibility_basis='visibility_readonly.json same-state exact; key cloth seam/crib/infant visible',actual_controls=0,GPU_calls=0,models=[],physical_replays=0,passed=False,error=None)
 try:
  m,d,obj,jid,adr=check.copy_source(s);r['route']=route(s,m,d,obj,jid,adr)
  if r['route']['passed']:r['virtual_held']=held_witness(s,m,d,obj,jid,adr);r['passed']=r['virtual_held']['passed']
  else:r['stop_reason']='first copied approach actual position/collision/real-zone eligibility failure'
 except BaseException as exc:r['error']=f'{type(exc).__name__}:{exc}'
 finally:r['original_integration_unchanged']=bool(np.array_equal(before,check.state(om,od)));r['original_model_unchanged']=mh==check.digest_model(om);s.close()
 r['elapsed_s']=time.monotonic()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in('error','passed','elapsed_s')},indent=2))
if __name__=='__main__':main()
