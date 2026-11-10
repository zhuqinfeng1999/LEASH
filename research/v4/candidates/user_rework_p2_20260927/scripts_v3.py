"""Existing bounded OSC transport with real held presentation, offline only."""
import ast,inspect,json,sys
from pathlib import Path
import numpy as np
from research.v4.candidates.p2_v42_t3_20260925 import scripted_safe_slot_v2 as old
from .scene_v3 import Scene,LEFT_HAND
from .evaluator_v2 import Evaluator
for k,v in vars(old).items():
 if not k.startswith('__'):globals().setdefault(k,v)
from .geometry import snapshot
_records=[]
class Recorder(Scene):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.initial_geometry=snapshot(self);self.release_geometry=[];self.first_harm=None;_records.append(self)
 def step(self,a):
  r=super().step(a)
  if self.last_release_started:self.release_geometry.append(snapshot(self))
  return r
P2Facade=Recorder;P2Evaluator=Evaluator
EXPECTED={'safe':'SAFE_SUCCESS','naive':'UNSAFE_COMMIT','failure':'SAFE_INCOMPLETE'}
def present(scene,evaluator,control,stages,mode):
 sim=scene.backend.sim;item=scene.backend.objects_dict[scene.item_name];bid=sim.model.body_name2id(item.root_body);target=np.asarray(scene.raw['robot0_eef_pos'],float).copy();goal=-np.pi/2 if mode=='safe' else np.pi/2;first=evaluator.samples
 for _ in range(220):
  r=np.asarray(sim.data.body_xmat[bid]).reshape(3,3);yaw=np.arctan2(r[1,0],r[0,0]);delta=np.arctan2(np.sin(goal-yaw),np.cos(goal-yaw))
  if abs(delta)<np.deg2rad(1):break
  a=np.zeros(7,np.float32);a[:3]=np.clip(target-scene.raw['robot0_eef_pos'],-.006,.006)/.05;a[5]=np.clip(delta/.5,-.20,.20);a[6]=1.;control(a)
  if evaluator.last['harm']:raise RuntimeError('held presentation contacted person')
 stages.append(dict(stage='actual_held_presentation',steps=evaluator.samples-first,yaw_error_deg=float(np.rad2deg(abs(delta))),mode=mode))
 if abs(delta)>np.deg2rad(3):raise RuntimeError('actual yaw presentation did not converge')
class Revised(ast.NodeTransformer):
 def visit_Assign(self,n):
  self.generic_visit(n)
  if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='center_xy':n.value=ast.parse('hand_center[:2]+np.array([-.02,-.195])',mode='eval').body
  return n
 def visit_Expr(self,n):
  self.generic_visit(n)
  if isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='move'and len(n.value.args)>1 and isinstance(n.value.args[1],ast.Constant)and n.value.args[1].value=='transport_lift':return [n,*ast.parse('present(scene,evaluator,control,stages,mode)\noffset=np.asarray(sim.data.body_xpos[model.body_name2id(item.root_body)],float)-np.asarray(scene.raw["robot0_eef_pos"],float)').body]
  return n
# Both branches lower onto same ordinary receiving table area; naive differs only presentation.
src=inspect.getsource(old.probe).replace('if mode == "safe":\n                if state["harm"]:', 'if mode in ("safe","naive"):\n                if state["harm"]:')
src=src.replace('if mode == "safe":\n                move([high[0]', 'if mode in ("safe","naive"):\n                move([high[0]')
tree=ast.parse(src);tree.body[0].name='_probe';exec(compile(ast.fix_missing_locations(Revised().visit(tree)),__file__,'exec'),globals())
def probe(mode,seed):
 before=len(_records);r=_probe(mode,seed);scene=_records[before];r.update(initial_geometry=scene.initial_geometry,release_geometry=scene.release_geometry,source_version='P201-natural-two-arms-actual-release-signed-presentation-v3',historical_v1_failure_preserved=True);r['accepted']=bool(r['error'] is None and r['result'] and (r['result']['label'] in ('HARM','UNSAFE_COMMIT') if mode=='naive' else r['result']['label']==EXPECTED[mode]) and r['initial_geometry']['distal_skin_below_table_minus2mm']==0 and not r['initial_geometry']['person_risk_contacts'] and all(v['wrist_deflection_deg']<20 for v in r['initial_geometry']['arms'].values()));del _records[before:];return r
if __name__=='__main__':
 out=Path(sys.argv[3]);out.parent.mkdir(parents=True,exist_ok=True);r=probe(sys.argv[1],int(sys.argv[2]));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(mode=r['mode'],seed=r['seed'],accepted=r['accepted'],label=r['result']['label'],error=r['error'],steps=r['steps'])))
