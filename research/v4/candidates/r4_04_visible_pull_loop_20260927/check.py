"""One copied full-solid approach/closure/held witness for visible loop."""
import copy,inspect,json,time,hashlib
import mujoco,numpy as np
from .scene import Scene,HERE,ROOT,POINT
from research.v4.candidates.r4_04_upright_pillow_fullsolid_z_20260927 import check as fullcheck
from research.v4.candidates.r4_04_visible_wood_riser_20260927 import check as wood
from scipy.spatial import ConvexHull

def route_fn():
 src=inspect.getsource(wood.parent.route);old="d.xmat[bid].reshape(3,3)@Rotation.from_euler('z',-np.pi/2).as_matrix()@np.asarray(old['SIDE_R'])";expression="np.stack([d.xmat[bid].reshape(3,3)[:,2],np.cross(np.array([0.,0.,-1.]),d.xmat[bid].reshape(3,3)[:,2]),np.array([0.,0.,-1.])],axis=1)";assert src.count(old)==1
 src=src.replace(old,expression).replace(" or row['rotation_residual_deg']>1",'')
 # The region now is actual visible upper bridge, so material containment must
 # use that bridge's physical vertices, not the unrelated main pillow hull.
 oldcloud="geom_points(m,d,m.geom('adult_pillow_1_pillow_body_collision').id)";assert src.count(oldcloud)==1
 src=src.replace(oldcloud,"geom_points(m,d,m.geom('adult_pillow_1_pull_loop_upper_bridge_collision').id)")
 ns=dict(vars(wood.parent));ns.update(POINT=POINT,cts=wood.contacts);exec(compile(src,__file__,'exec'),ns);return ns['route']

def held_fn():
 src=inspect.getsource(wood.held.held_witness)
 old="check.geom_points(m,d,m.geom('adult_pillow_1_pillow_body_collision').id)";assert src.count(old)==1
 src=src.replace(old,"check.geom_points(m,d,m.geom('adult_pillow_1_pull_loop_upper_bridge_collision').id)").replace('newgap=hi-lo+.002','newgap=2*max(abs(lo),abs(hi))+.001')
 ns=dict(vars(wood.held));ns['check']=__import__('types').SimpleNamespace(**{**vars(wood.parent),'cts':wood.contacts});exec(compile(src,__file__,'exec'),ns);return ns['held_witness']

def main():
 out=HERE/'report.json'
 if out.exists():raise RuntimeError('First loop candidate immutable')
 start=time.monotonic();r=dict(kind='ONE_VISIBLE_NORMAL_LOOP_NEW_AFFORDANCE_COPY_ONLY',physical_controls=0,GPU=False,models=[],passed=False,error=None)
 pins=json.loads((ROOT/'research/v4/candidates/r4_04_upright_pillow_fullsolid_z_20260927/source_freeze.json').read_text());r['parent_source_drift']=[k for k,v in pins.items()if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
 if r['parent_source_drift']:raise RuntimeError('Parent source drift')
 s=Scene(0);om,od=s.backend.sim.model._model,s.backend.sim.data._data;before=wood.parent.state(om,od);mh=wood.parent.digest_model(om)
 try:
  r['visible_loop']=s.visible_loop_metadata;r['scene_initial']=s.upright_initial;m=copy.deepcopy(om);d=mujoco.MjData(m);mujoco.mj_setState(m,d,before,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);obj=s.backend.objects_dict[s.item_name];jid=m.joint(obj.joints[0]).id;adr=m.jnt_qposadr[jid]
  # Shared initial routine keys the actual site; Scene supplies its new visible annotation.
  r['initial']=fullcheck.initial(s,m,d,obj)
  if not r['initial']['passed']:r['stop_reason']='first actual initial support/contact/public-key failure'
  else:
   r['route']=route_fn()(s,m,d,obj,jid,adr)
   if not r['route']['passed']:r['stop_reason']='first copied approach/palm/finger/material/region failure'
   else:r['close_and_virtual_held']=held_fn()(s,m,d,obj,jid,adr);r['passed']=bool(r['close_and_virtual_held']['passed']);r['stop_reason']=None if r['passed']else'first copied closure/held reach/contact/critical failure'
 except BaseException as exc:r['error']=f'{type(exc).__name__}: {exc}'
 finally:r['original_integration_unchanged']=bool(np.array_equal(before,wood.parent.state(om,od)));r['original_model_unchanged']=mh==wood.parent.digest_model(om);s.close()
 r['elapsed_s']=time.monotonic()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r.get(k)for k in('passed','error','stop_reason','elapsed_s')}))
if __name__=='__main__':main()
