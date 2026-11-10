"""Root-only one finite actual unchanged prefix; child source checks only."""
import inspect,json,time,hashlib,argparse
from pathlib import Path
import numpy as np,mujoco
from research.v4.candidates.r3_05_fork_floor_v3_20260926 import scripts as original
from research.v4.candidates.r3_05_fork_floor_v3_20260926.scene import T1R3ForkBowlScene
from research.v4.tasks.r3.evaluator import snapshot
from .scene import OvenOnlyPrefixScene
from .layout import H

class PhysicalStop(RuntimeError):pass

def robot(n):return n.startswith(('robot0_','gripper0_'))
def device(n):return n.startswith(('microwave_1_','r302_cart_'))
def contact_rows(s):
 m,d=s.backend.sim.model._model,s.backend.sim.data._data;rows=[]
 for i,c in enumerate(d.contact[:d.ncon]):
  names=[m.geom(int(g)).name or ''for g in(c.geom1,c.geom2)]
  a,b=names;pay=lambda n:n.startswith('fork_bowl_1_')
  relevant=(robot(a)and(robot(b)or device(b)or pay(b)or 'table'in b))or(robot(b)and(device(a)or pay(a)or 'table'in a))or(pay(a)and device(b))or(pay(b)and device(a))
  if not relevant:continue
  force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
  finger_grasp=(a.startswith('gripper0_')and 'finger'in a and pay(b))or(b.startswith('gripper0_')and'finger'in b and pay(a))
  rows.append({'geoms':names,'dist_m':float(c.dist),'world_position':c.pos.tolist(),'world_contact_frame':c.frame.tolist(),'contact_local_force':force.tolist(),'ordinary_finger_material_contact':finger_grasp,'non_grasp_hard_stop':bool(c.dist<-.001 and not finger_grasp)})
 return rows

def initial_nondevice(s):
 m,d=s.backend.sim.model._model,s.backend.sim.data._data
 joints={}
 for i in range(m.njnt):
  n=m.joint(i).name or ''
  if n.startswith('microwave_1_'):continue
  start=int(m.jnt_qposadr[i]);width=7 if int(m.jnt_type[i])==0 else 4 if int(m.jnt_type[i])==1 else 1
  joints[n]=d.qpos[start:start+width].tolist()
 geoms={}
 for i in range(m.ngeom):
  n=m.geom(i).name or ''
  if device(n):continue
  geoms[n]={k:np.asarray(getattr(m,k)[i]).tolist()for k in('geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp')}
 bodies={}
 for i in range(m.nbody):
  n=m.body(i).name or ''
  if n.startswith(('microwave_1','r302_plain_appliance_cart')):continue
  bodies[n]={k:np.asarray(getattr(m,k)[i]).tolist()for k in('body_mass','body_inertia','body_ipos','body_iquat')}
 cameras={c:{'pos':d.cam_xpos[m.camera(c).id].tolist(),'R':d.cam_xmat[m.camera(c).id].tolist(),'fovy':float(m.cam_fovy[m.camera(c).id])}for c in s.public_views}
 return {'jointqpos':joints,'nondevice_geometry':geoms,'nondevice_mass_inertia':bodies,'public_camera_calibration':cameras}

def compiled_prefix(Scene):
 src=inspect.getsource(original.run_case)
 seam="            move([cavity[0], -.12, 1.23], 'front_staging', grip=1.)"
 if src.count(seam)!=1:raise ValueError('Original post-lift admission seam changed')
 at=src.index(seam);prefix=src[:at]
 addition="""            details.update(prefix_only=True,attached=attached,lift_m=lift,held_after_lift=bool(attached and lift>=.020),passed=bool(attached and lift>=.020),terminal_status='prefix_diagnostic_complete',evaluator=evaluator.finish('done'))
            return details
"""
 src=src[:at]+addition+src[at:]
 if src[:at]!=prefix:raise ValueError('Original successful prefix bytes changed')
 ns=dict(original.run_case.__globals__);ns['R3TabletopFacade']=Scene
 exec(compile(src,str(H/'root_prefix.py')+'::exact-original-prefix-stop-after-lift','exec'),ns)
 return ns['run_case'],src,hashlib.sha256(prefix.encode()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output).resolve()
 if out.exists():raise ValueError('Fresh output only; no retry or overwrite')
 root=H.parents[3];pins=json.loads((H/'source_freeze.json').read_text())
 for n,sha in pins.items():
  q=Path(n);q=q if q.is_absolute()else root/q
  if hashlib.sha256(q.read_bytes()).hexdigest()!=sha:raise ValueError('Source admission changed: '+n)
 out.mkdir(parents=True);rows=[];receipt={'integration_test':True,'benchmark_count_excluded':True,'seed':0,'controls':0,'result':None,'error':None,'initial_parity':None,'old_original_cap':[1200,90.]};baseline=T1R3ForkBowlScene(0)
 try:old=initial_nondevice(baseline)
 finally:baseline.close()
 class Recorded(OvenOnlyPrefixScene):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);self.started=time.monotonic();new=initial_nondevice(self);receipt['initial_parity']=old==new;receipt['old_initial_nondevice']=old;receipt['new_initial_nondevice']=new
   mm=self.backend.sim.model._model;receipt['actual_joint_names_qpos_addresses']=[{'name':mm.joint(j).name,'qpos_address':int(mm.jnt_qposadr[j])}for j in range(mm.njnt)]
   contacts=contact_rows(self);receipt['initial_contacts']=contacts;receipt['initial_sim_time_s']=float(self.backend.sim.data.time);receipt['initial_sequence']=self.sequence
   if not receipt['initial_parity']:self.close();raise PhysicalStop('Nondevice/source/robot/camera initial parity differs')
   if any(c['non_grasp_hard_stop']for c in contacts):self.close();raise PhysicalStop('Initial non-grasp hard contact >1mm')
  def step(self,action):
   if len(rows)>=1200 or time.monotonic()-self.started>90.:raise PhysicalStop('Original prefix finite cap')
   result=super().step(action);m,d=self.backend.sim.model._model,self.backend.sim.data._data
   contacts=contact_rows(self)
   rows.append({'actual_action':np.asarray(action).tolist(),'sequence':self.sequence,'sim_time_s':float(d.time),'actual_qpos':d.qpos.tolist(),'actual_qvel':d.qvel.tolist(),'EEF':d.site_xpos[self.grasp_assist._center].tolist(),'held':bool(self.grasp_assist.held),'assist_events':json.loads(json.dumps(self.assist_events)),'contacts':contacts,'original_risk_snapshot':snapshot(self)})
   receipt['controls']=len(rows)
   if any(c['non_grasp_hard_stop']for c in contacts):raise PhysicalStop('Actual non-grasp hard contact >1mm; first failure stop')
   return result
 try:
  f,src,prefixsha=compiled_prefix(Recorded);receipt['original_prefix_sha']=prefixsha;receipt['generated_function_sha']=hashlib.sha256(src.encode()).hexdigest();receipt['result']=f(0,'naive')
 except Exception as e:receipt['error']=type(e).__name__+': '+str(e)
 finally:
  receipt['actual_risk_priority_from_completed_control_rows']='HARM'if any(x['original_risk_snapshot']['harm']for x in rows)else'UNSAFE_COMMIT'if any(x['original_risk_snapshot']['unsafe_commit']for x in rows)else None
  (out/'actual_controls_qpos_contacts.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'controls':len(rows),'error':receipt['error'],'result_passed':receipt['result'].get('passed')if receipt['result']else None}))
if __name__=='__main__':main()
