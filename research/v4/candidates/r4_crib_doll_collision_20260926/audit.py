"""Offline-only compiled doll collision audit; old labels remain immutable."""
import hashlib,json,time
from pathlib import Path
from unittest.mock import patch
import mujoco,numpy as np
from research.v4.candidates.r4_02_plug_surface_v2_20260926 import scene as cable, scripts as cable_scripts
from research.v4.candidates.r4_05_identity_uv_fix_20260926 import scene as bead
from research.v4.candidates.r4_05_live_rim_grasp_20260926 import scripts as bead_scripts
from research.v4.candidates.r4_06_flat_underbase_v2_20260926 import scene as coin,scripts as coin_scripts
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROVIDERS={'R4-02':(cable.T1R4CableScene,cable_scripts,'T1R4CableScene'),'R4-05':(bead.ReadableIdentityScene,bead_scripts,'T1R4MagneticBoxScene'),'R4-06':(coin.T1R4CoinDishScene,coin_scripts,'T1R4CoinDishScene')}
def compiled(s):
 sim=s.backend.sim;m,d=sim.model._model,sim.data._data
 dolls=[i for i in range(m.ngeom)if '_doll_'in(m.geom(i).name or'')];cribs=[i for i in range(m.ngeom)if(m.geom(i).name or'').startswith('crib_1_')];item=[i for i in range(m.ngeom)if(m.geom(i).name or'').startswith(s.item_name+'_')and(m.geom_contype[i]or m.geom_conaffinity[i])]
 rows=[dict(name=m.geom(i).name,id=i,type=int(m.geom_type[i]),size=m.geom_size[i].tolist(),local_pos=m.geom_pos[i].tolist(),local_quat=m.geom_quat[i].tolist(),body_id=int(m.geom_bodyid[i]),contype=int(m.geom_contype[i]),conaffinity=int(m.geom_conaffinity[i]),group=int(m.geom_group[i]))for i in dolls]
 # Search every actual crib geom, not just the named doll shapes; no separate
 # enabled body collider can be inferred from visual-only names.
 return dict(doll_geoms=rows,enabled_doll_geoms=[r['name']for r in rows if r['contype']or r['conaffinity']],all_enabled_crib_geoms=[m.geom(i).name for i in cribs if m.geom_contype[i]or m.geom_conaffinity[i]],item_collision_geoms=[m.geom(i).name for i in item]),dolls,item
class OverlapTracker:
 def __init__(self):self.first=None;self.worst=None;self.final=None;self.frames=0;self.compiled=None;self.true_contacts=[]
 def observe(self,s):
  sim=s.backend.sim;m,d=sim.model._model,sim.data._data
  if self.compiled is None:self.compiled,self.dolls,self.items=compiled(s)
  best=None
  for a in self.items:
   for b in self.dolls:
    dist=float(mujoco.mj_geomDistance(m,d,a,b,.05,None))
    if dist<-.0001 and(best is None or dist<best['distance_m']):best=dict(item=m.geom(a).name,doll=m.geom(b).name,distance_m=dist,sim_time_s=float(d.time),held=bool(s.grasp_assist.held),item_qpos=sim.data.get_joint_qpos(s.backend.objects_dict[s.item_name].joints[0]).copy().tolist())
  self.frames+=1;self.final=best
  if best is not None:
   if self.first is None:self.first=best
   if self.worst is None or best['distance_m']<self.worst['distance_m']:self.worst=best
  for c in d.contact[:d.ncon]:
   if(int(c.geom1)in self.items and int(c.geom2)in self.dolls)or(int(c.geom2)in self.items and int(c.geom1)in self.dolls):self.true_contacts.append(dict(geoms=[m.geom(int(i)).name for i in(c.geom1,c.geom2)],distance_m=float(c.dist),sim_time_s=float(d.time)))
 def result(self):return dict(compiled=self.compiled,frames=self.frames,first_geometrical_overlap=self.first,worst_geometrical_overlap=self.worst,final_geometrical_overlap=self.final,actual_payload_doll_contacts=self.true_contacts)
def run_old(instance):
 cls,scripts,token=PROVIDERS[instance];tracker=OverlapTracker()
 class Observed(cls):
  def __init__(self,*a,**kw):super().__init__(*a,**kw);tracker.observe(self)
  def step(self,a):r=super().step(a);tracker.observe(self);return r
  def close(self):tracker.observe(self);super().close()
 scope=patch.dict(scripts.namespace,{token:Observed})if hasattr(scripts,'namespace')else patch.object(scripts,token,Observed)
 with scope:r=scripts.run_case(0,'naive')
 return dict(instance=instance,kind='Readonly old-source CPU trajectory; not requalification or relabel',provider_file=str(Path(cls.__module__.replace('.','/')).with_suffix('.py')),old_script_result=r,doll_collision_evidence=tracker.result())
def main():
 p=HERE/'old_compiled_trajectory_audit.json'
 if p.exists():raise ValueError('Preserve prior audit')
 start=time.monotonic();report=dict(shared_crib_source_sha256=hashlib.sha256((ROOT/'research/v4/tasks/r4/crib.xml').read_bytes()).hexdigest(),model_calls=[],gpu_used=False,rows=[])
 try:
  for instance in PROVIDERS:
   row=run_old(instance);report['rows'].append(row);e=row['doll_collision_evidence'];print(json.dumps(dict(instance=instance,enabled_doll=len(e['compiled']['enabled_doll_geoms']),worst_overlap=e['worst_geometrical_overlap'],true_doll_contacts=len(e['actual_payload_doll_contacts']))),flush=True)
 finally:report['elapsed_s']=time.monotonic()-start;p.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
