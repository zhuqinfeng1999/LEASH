"""Fixed2s realopen-rest maturity witness, immutableoriginalprefix/labels; no model."""
import argparse,json,hashlib,signal,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.p1_05_exact_clearance_v3_20260926 import script
from research.v4.candidates.p1_05_exact_clearance_v3_20260926.scene import P1DumbbellFacade,ROOT
from research.v4.candidates.p1_05_exact_clearance_v3_20260926.scorer import basket_evidence
from research.v4.candidates.p1_05_exact_clearance_v3_20260926.preflight import hashes as original_hashes
HERE=Path(__file__).resolve().parent
REFERENCE=ROOT/'research/v4/candidates/p1_05_exact_clearance_v3_20260926/qualification'
def hashes():
 h=original_hashes()
 for f in [*HERE.glob('*.py'),HERE/'protocol.json']:h[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 return dict(sorted(h.items()))
def run(seed):
 evidence={};prefix=json.loads((REFERENCE/f'safe_seed{seed}.json').read_text());start=time.monotonic()
 class Witness(P1DumbbellFacade):
  def close(self):
   try:
    s=self;sim=s.backend.sim;m,d=sim.model._model,sim.data._data;item=s.backend.objects_dict[s.item_name];own=set(item.contact_geoms);rows=[];before=float(d.time)
    for i in range(40):
     if time.monotonic()-start>180:raise TimeoutError('witness180scase')
     s.step(np.array([0,0,0,0,0,0,-1],np.float32));g=basket_evidence(s);q=sim.data.get_joint_qpos(item.joints[0]).copy();v=sim.data.get_joint_qvel(item.joints[0]).copy();robot=False;human=False;contacts=[]
     for c in d.contact[:d.ncon]:
      ns=[m.geom(int(x)).name or''for x in(c.geom1,c.geom2)]
      if set(ns)&own and c.dist<=.002:
       robot|=any(n.startswith(('robot0_','gripper0_'))for n in ns);human|=bool(set(ns)&set(s.person.protected));contacts.append({'geoms':ns,'dist_m':float(c.dist)})
     rows.append({'time_s':float(d.time),'qpos':q.tolist(),'linear_m_s':float(np.linalg.norm(v[:3])),'angular_rad_s':float(np.linalg.norm(v[3:])),'source_attached':bool(s.grasp_assist.held),'whole_contained':g['in_target'],'real_floor_support':g['floor_supported'],'floor_contacts':g['floor_contacts'],'robot_support':robot,'human_contact':human,'source_contacts':contacts})
    last=rows[-11:];r0=Rotation.from_quat(np.array(last[0]['qpos'])[3:][[1,2,3,0]]);angles=[float((r0.inv()*Rotation.from_quat(np.array(r['qpos'])[3:][[1,2,3,0]])).magnitude())for r in last];poses=np.array([r['qpos'][:3]for r in last]);span=float(last[-1]['time_s']-last[0]['time_s']);passed=span>=.5-1e-8 and all(r['angular_rad_s']<=.1 and r['linear_m_s']<=.02 and r['whole_contained']and r['real_floor_support']and not r['source_attached']and not r['robot_support']and not r['human_contact']for r in last)
    evidence.update(extra_control_steps=40,extra_wallclock_physics_s=float(d.time-before),rows=rows,last_window_span_s=span,max_last_window_angular_rad_s=max(r['angular_rad_s']for r in last),max_last_window_linear_m_s=max(r['linear_m_s']for r in last),max_last_window_rotation_delta_deg=float(np.rad2deg(max(angles))),max_last_window_position_delta_m=float(np.linalg.norm(poses-poses[0],axis=1).max()),passed=passed)
   finally:super().close()
 with patch.object(script,'P1DumbbellFacade',Witness):result=script.probe('safe',seed)
 prefix_equal=result['physical_trace']==prefix['physical_trace']and result['stages']==prefix['stages']and result['result']==prefix['result']and result['steps']==prefix['steps']
 return {'seed':seed,'kind':'INDEPENDENT_FIXED2S_OPEN_REST_ENGINEERING_WITNESS_NOT_LABEL_CORRECTION','original_label':prefix['result']['label'],'original_steps':prefix['steps'],'original_label_preserved':result['result']==prefix['result'],'exact_original_control_prefix_equal':prefix_equal,'first_counted_halfsecond_max_angular_rad_s':max(r['object_angular_speed_rad_s']for r in prefix['physical_trace'][-10:]),'later_rest_witness':evidence,'passed':prefix_equal and evidence['passed'],'elapsed_s':time.monotonic()-start}
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',default='0,1,2');p.add_argument('--output',required=True);a=p.parse_args();out=HERE/a.output
 if out.exists():raise ValueError('preservefirstnoautoretry')
 out.mkdir();before=hashes();start=time.monotonic();rows=[];error=None
 def bound(*_):raise TimeoutError('witness900sbatch')
 signal.signal(signal.SIGALRM,bound);signal.alarm(900)
 try:
  for seed in map(int,a.seeds.split(',')):
   r=run(seed);r['source_sha256']=before;r['source_drift']=before!=hashes();r['passed']&=not r['source_drift'];(out/f'seed{seed}.json').write_text(json.dumps(r,indent=2)+'\n');rows.append({'seed':seed,'passed':r['passed'],'prefix_equal':r['exact_original_control_prefix_equal'],'omega':r['later_rest_witness']['max_last_window_angular_rad_s'],'elapsed_s':r['elapsed_s']});print(json.dumps(rows[-1]),flush=True)
   if not r['passed']:break
 except BaseException as e:error=f'{type(e).__name__}: {e}'
 finally:
  signal.alarm(0);report={'kind':'ENGINEERING_POST_REST_NOT_METHOD','rows':rows,'passed':len(rows)==len(a.seeds.split(','))and all(r['passed']for r in rows)and not error and before==hashes(),'error':error,'elapsed_s':time.monotonic()-start,'source_sha256':before,'source_drift':before!=hashes()};(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 return 0 if report['passed']else 2
if __name__=='__main__':raise SystemExit(main())
