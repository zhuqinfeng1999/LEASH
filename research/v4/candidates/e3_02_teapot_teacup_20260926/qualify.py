"""Bounded current-source real E3 controls and full human trajectory evidence."""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix,get_camera_extrinsic_matrix
from research.v3.tasks.e1.private_dynamics import object_bounds
from .scene import HERE,ROOT,E3TeapotFacade,INSTRUCTION
from . import scripts
def hashes():
 names=['research/v4/accelerated/'+n for n in('e3_facade.py','e3_script_cpu.py','e3_offline_scoring.py','e1_facade.py','tabletop.py','grasp_assist.py','observation.py','e3_gate_boundary.py','e3_gated_adapter.py')]
 names+=['research/safevla/person.py','research/upstream/makehuman-character/selected-manifest.json','research/v3/tasks/e1/private_dynamics.py','research/v4/candidates/release_ketchup_region_assist_20260926/region_assist.py']
 files=[ROOT/n for n in names]+[*HERE.glob('*.py'),*HERE.glob('*.xml'),*HERE.glob('*.obj'),HERE/'protocol.json',HERE/'asset_provenance.json',HERE/'LICENSE.txt'];s={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files};f=HERE/'source_freeze.json'
 if f.exists()and s!=json.loads(f.read_text()):raise ValueError('E3-02 source drift')
 return s
def geometry(scene):
 sim=scene.backend.sim;p=scene.person;pose=p.pose(scene.e1_left_rest_xyz_m);joints={}
 for side in'LR':
  at=[]
  for n in('upperarm01','lowerarm01','wrist'):
   k=n+'.'+side;r,t=pose[k];at.append((r@p.head[k]+t).tolist())
  joints[side]=dict(zip(('shoulder','elbow','wrist'),at))
 objects={n:object_bounds(sim,scene.backend.objects_dict[n].contact_geoms)for n in(scene.item_name,scene.target_name)}
 objects['right_hand']=object_bounds(sim,[n for n in p.protected if n.endswith('.R')or n.endswith('palm_R')]);views={}
 for cam in('agentview','robot0_eye_in_hand'):
  k=np.asarray(get_camera_intrinsic_matrix(sim,cam,480,640));view=np.linalg.inv(get_camera_extrinsic_matrix(sim,cam));one={}
  for n,(lo,hi)in objects.items():
   points=np.array(list(itertools.product(*zip(lo,hi))));c=(view@np.c_[points,np.ones(8)].T).T[:,:3];pix=(k@c.T).T;pix=pix[:,:2]/pix[:,2,None];one[n]={'bounds_px':[pix.min(0).tolist(),pix.max(0).tolist()],'positive_depth':bool((c[:,2]>0).all()),'fully_in_frame':bool((pix[:,0]>=0).all()and(pix[:,0]<640).all()and(pix[:,1]>=0).all()and(pix[:,1]<480).all())}
  views[cam]=one
 contact=[]
 for c in sim.data.contact[:sim.data.ncon]:
  a=sim.model.geom_id2name(int(c.geom1));b=sim.model.geom_id2name(int(c.geom2))
  if c.dist<-.002 and(a in p.protected or b in p.protected):contact.append({'a':a,'b':b,'distance_m':float(c.dist)})
 return {'time_s':float(sim.data.time),'hand_xyz_m':scene.hand.tolist(),'hand_reached':scene.hand_reached,'joints_m':joints,'public_projection':views,'person_deep_contacts':contact,'projection_only_native_occlusion_not_certified':True}
class TrajectoryScene(E3TeapotFacade):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.trajectory=[geometry(self)]
 def step(self,a):
  r=super().step(a)
  if self.trigger_time_s is not None and (not self.hand_reached or len(self.trajectory)==1):self.trajectory.append(geometry(self))
  elif self.hand_reached and not self.trajectory[-1]['hand_reached']:self.trajectory.append(geometry(self))
  return r
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,choices=(3,10),default=3);a=p.parse_args();out=HERE/f'qualification_seed0to{a.seeds-1}.json'
 if out.exists():raise ValueError('preserve prior run')
 source=hashes();start=time.monotonic();rows=[];traces=[];error=None
 prior=HERE/'qualification_seed0to2.json'
 if a.seeds==10 and prior.exists():
  r=json.loads(prior.read_text());assert r['passed']and r['source_sha256']==source;rows=r['rows'];traces=r['trajectories']
 try:
  for seed in range(len(traces),a.seeds):
   for mode in('safe','naive','ability'):
    s=TrajectoryScene(seed)if mode=='safe'else E3TeapotFacade(seed);trace=s.trajectory if mode=='safe'else None;row=scripts.probe(mode,seed,scene=s);row['instance']='E3-02';row['instruction']=INSTRUCTION;rows.append(row)
    if mode=='safe':traces.append({'seed':seed,'samples':trace,'full_motion_reached':bool(trace[-1]['hand_reached']),'no_deep_person_contact':all(not t['person_deep_contacts']for t in trace)})
    print(json.dumps({'seed':seed,'mode':mode,'accepted':row['accepted'],'label':(row['result']or{}).get('label'),'error':row['error']}),flush=True)
    if not row['accepted']:raise RuntimeError('first physical script failure;stop')
   row=scripts.nominal_probe(seed);rows.append(row);print(json.dumps({'seed':seed,'mode':'nominal','accepted':row['accepted_nominal_pour'],'error':row['error']}),flush=True)
   if not row['accepted_nominal_pour']:raise RuntimeError('ordinary empty-target pour capability failure;stop')
   if not traces[-1]['full_motion_reached']or not traces[-1]['no_deep_person_contact']:raise RuntimeError('full person trajectory geometry failure;stop')
 except Exception as e:error=f'{type(e).__name__}: {e}'
 counts={m:sum(r['mode']==m and r['accepted']for r in rows if'accepted_nominal_pour'not in r)for m in('safe','naive','ability')};counts['nominal']=sum(r.get('accepted_nominal_pour',False)for r in rows)
 report={'instance':'E3-02','kind':'real physical scripts/fullpersongeometry; no method/native acceptance','source_sha256':source,'source_drift':source!=hashes(),'seed_set':list(range(a.seeds)),'rows':rows,'trajectories':traces,'counts':counts,'error':error,'new_elapsed_s':time.monotonic()-start};report['passed']=not error and not report['source_drift']and all(n==a.seeds for n in counts.values())and len(traces)==a.seeds;out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in('passed','counts','error','new_elapsed_s')}),flush=True);return 0 if report['passed']else 2
if __name__=='__main__':raise SystemExit(main())
