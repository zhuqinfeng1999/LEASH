"""Same loop scene, legal copied close; raw overclose version immutable."""
import inspect,json,copy,time
from pathlib import Path
import mujoco,numpy as np
from research.v4.candidates.r4_04_visible_pull_loop_20260927 import check as prev
from research.v4.candidates.r4_04_visible_pull_loop_20260927.scene import Scene
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]

def held_fn():
 src=inspect.getsource(prev.wood.held.held_witness);needle="check.geom_points(m,d,m.geom('adult_pillow_1_pillow_body_collision').id)";assert src.count(needle)==1
 src=src.replace(needle,"check.geom_points(m,d,m.geom('adult_pillow_1_pull_loop_upper_bridge_collision').id)").replace('newgap=hi-lo+.002','newgap=2*max(abs(lo),abs(hi))+.001')
 anchor=" for t in np.linspace(0,1,6)[1:]:"
 added=" r['unclipped_helper_close_qpos']=end.tolist();end=np.array([np.clip(end[n],m.jnt_range[i,0],m.jnt_range[i,1])for n,i in enumerate(finger)]);r['legal_close_qpos']=end.tolist();r['actual_finger_limits']=[m.jnt_range[i].tolist()for i in finger]\n"+anchor
 assert src.count(anchor)==1;src=src.replace(anchor,added)
 bad="bad=[x for x in cts if x['robot']and x['distance_m']<-.001 and(x['pillow']or x['table']or x['infant'])]"
 fixed="bad=[x for x in cts if x['robot']and x['distance_m']<-.001 and(x['pillow']or x['table']or x['infant']or any(z.startswith('fixed_mount')for z in x['pair']))and not(any(z.startswith('adult_pillow_1_pull_loop_')for z in x['pair'])and any(z.startswith('gripper0_')and 'finger'in z for z in x['pair']))]"
 assert src.count(bad)==1;src=src.replace(bad,fixed)
 ns=dict(vars(prev.wood.held));ns['check']=__import__('types').SimpleNamespace(**{**vars(prev.wood.parent),'cts':prev.wood.contacts});exec(compile(src,__file__,'exec'),ns);return ns['held_witness']

def main():
 out=HERE/'report.json'
 if out.exists():raise RuntimeError('One legal-close continuation only')
 st=time.monotonic();s=Scene(0);om,od=s.backend.sim.model._model,s.backend.sim.data._data;before=prev.wood.parent.state(om,od);mh=prev.wood.parent.digest_model(om);r=dict(physical_controls=0,GPU=False,models=[],passed=False,error=None)
 try:
  m=copy.deepcopy(om);d=mujoco.MjData(m);mujoco.mj_setState(m,d,before,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);obj=s.backend.objects_dict[s.item_name];jid=m.joint(obj.joints[0]).id;adr=m.jnt_qposadr[jid];r['initial']=prev.fullcheck.initial(s,m,d,obj);r['route']=prev.route_fn()(s,m,d,obj,jid,adr)
  if not r['initial']['passed']or not r['route']['passed']:r['stop_reason']='Unexpected current original prefix failure'
  else:r['legal_close_and_virtual_held']=held_fn()(s,m,d,obj,jid,adr);r['passed']=bool(r['legal_close_and_virtual_held']['passed'])
 except BaseException as exc:r['error']=f'{type(exc).__name__}: {exc}'
 finally:r['original_integration_unchanged']=bool(np.array_equal(before,prev.wood.parent.state(om,od)));r['original_model_unchanged']=mh==prev.wood.parent.digest_model(om);s.close()
 r['elapsed_s']=time.monotonic()-st;out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r.get(k)for k in('passed','error','stop_reason','elapsed_s')}))
if __name__=='__main__':main()
