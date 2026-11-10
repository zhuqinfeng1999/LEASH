"""ONE root-authorized seed0 original safe+fixed2s rest, MULTICCD bit only."""
import hashlib,json,signal,time,numbers
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
from research.v4.candidates.p1_05_post_rest_witness_20260926 import driver
from research.v4.candidates.p1_05_exact_clearance_v3_20260926.scene import P1DumbbellFacade
from research.v4.candidates.p1_05_exact_clearance_v3_20260926.scorer import P1BasketEvaluator,basket_evidence
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
CAP=1172;WALL=180

def digest_array(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def model_arrays(m):
 return {k:digest_array(getattr(m,k))for k in dir(m)if not k.startswith('_')and isinstance(getattr(m,k),np.ndarray)}
def option_values(opt):
 out={}
 for k in dir(opt):
  if k.startswith('_'):continue
  v=getattr(opt,k)
  if isinstance(v,np.ndarray):out[k]=v.tolist()
  elif isinstance(v,numbers.Number):out[k]=float(v)if isinstance(v,float)else int(v)
 return out

def main():
 out=HERE/'seed0_trial'
 if out.exists():raise RuntimeError('exactly one fresh trial; no retry')
 pins=json.loads((HERE/'source_freeze.json').read_text())['source_sha256']
 drift=[p for p,v in pins.items()if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=v]
 if drift:raise RuntimeError(f'source drift before launch:{drift}')
 out.mkdir();started=time.monotonic();rows=0;diag={};original=None;error=None;ref=[]
 (out/'started.json').write_text(json.dumps(dict(seed=0,controls_cap=CAP,wall_cap_s=WALL,models=[],gpu=False,approved_change='only MULTICCD enablebit16',source_freeze_sha256=hashlib.sha256((HERE/'source_freeze.json').read_bytes()).hexdigest()),indent=2)+'\n')
 trace=(out/'actual_control_states.jsonl').open('w')
 class Scene(P1DumbbellFacade):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);self._steps=0;self._private_evaluator=P1BasketEvaluator();m=self.backend.sim.model._model
   before_arrays=model_arrays(m);before_options=option_values(m.opt);old=int(m.opt.enableflags);m.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
   after_arrays=model_arrays(m);after_options=option_values(m.opt)
   changes={k:[before_options[k],after_options[k]]for k in before_options if before_options[k]!=after_options[k]}
   if before_arrays!=after_arrays or changes!={'enableflags':[0,16]}:super().close();raise RuntimeError(f'only enableflags0->16 allowed:{changes}')
   diag.update(model_arrays_equal=before_arrays==after_arrays,option_changes=changes,source_mass_kg=float(m.body_mass[self.backend.sim.model.body_name2id(self.backend.objects_dict[self.item_name].root_body)]),source_inertia=m.body_inertia[self.backend.sim.model.body_name2id(self.backend.objects_dict[self.item_name].root_body)].tolist(),original_model_array_sha256=before_arrays,original_options=before_options,new_options=after_options,mujoco_version=mujoco.__version__,control_protocol='Original init/park retained; flag enabled before first scripted control; common all-method Scene protocol')
   self._private_evaluator.observe(self);ref.append(self)
  def step(self,action):
   nonlocal rows
   if self._steps>=CAP:raise TimeoutError('declared1172controlcap')
   if time.monotonic()-started>WALL:raise TimeoutError('declared180swallcap')
   result=super().step(action);self._steps+=1;rows+=1;self._private_evaluator.observe(self)
   sim=self.backend.sim;m,d=sim.model._model,sim.data._data;contacts=[];item=self.backend.objects_dict[self.item_name];own=set(item.contact_geoms)
   for i,c in enumerate(d.contact[:d.ncon]):
    names=[m.geom(int(g)).name or''for g in(c.geom1,c.geom2)]
    if not own&set(names):continue
    force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
    contacts.append(dict(pair=names,distance_m=float(c.dist),position_m=c.pos.tolist(),frame_world=c.frame.reshape(3,3).tolist(),contact_frame_force=force.tolist(),solref=c.solref.tolist(),solimp=c.solimp.tolist(),friction=c.friction.tolist(),dim=int(c.dim)))
   integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION)
   q=sim.data.get_joint_qpos(item.joints[0]).copy();v=sim.data.get_joint_qvel(item.joints[0]).copy();g=basket_evidence(self)
   trace.write(json.dumps(dict(sequence=self._steps,sim_time_s=float(d.time),control=np.asarray(action).tolist(),source_qpos=q.tolist(),source_qvel=v.tolist(),angular_speed_rad_s=float(np.linalg.norm(v[3:])),linear_speed_m_s=float(np.linalg.norm(v[:3])),integration_state=integration.tolist(),source_contacts=contacts,basket_evidence=g,private_harm=bool(self._private_evaluator.first_harm_s is not None),held=bool(self.grasp_assist.held),solver_niter=np.asarray(d.solver_niter).tolist()))+'\n');trace.flush()
   return result
 def bound(*_):raise TimeoutError('declared180swallcap')
 signal.signal(signal.SIGALRM,bound);signal.alarm(WALL)
 try:
  with patch.object(driver,'P1DumbbellFacade',Scene):original=driver.run(0)
 except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
 finally:
  signal.alarm(0);trace.close()
  current_drift=[p for p,v in pins.items()if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=v]
  last_score=ref[0]._private_evaluator.finish('done')if ref else None
  passed=bool(original and original['passed']and not error and not current_drift and rows<=CAP)
  label='HARM'if last_score and last_score['label']=='HARM'else('SAFE_SUCCESS'if passed else'SAFE_INCOMPLETE')
  if error and not isinstance(error,str):error=str(error)
  system_failure=bool(error and not error.startswith('TimeoutError:'))
  report=dict(kind='ONE_SEED0_MULTICCD_PHYSICAL_DIAGNOSTIC_NOT_METHOD_RESULT',seed=0,passed=passed,engineering_label=label,system_failure=system_failure,error=error,actual_controls=rows,controls_cap=CAP,wall_cap_s=WALL,elapsed_s=time.monotonic()-started,source_drift=current_drift,model_change=diag,original_safe_and_fixed2s_rest=original,private_final_score=last_score,models=[],gpu=False,no_old_label_changed=True)
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(passed=report['passed'],controls=rows,error=error,label=label,rest_passed=(original or{}).get('later_rest_witness',{}).get('passed'))));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
