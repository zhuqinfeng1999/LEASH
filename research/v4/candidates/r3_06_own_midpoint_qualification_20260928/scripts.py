"""Thin original branch/control binding; new own seeded dynamic tail only."""
import ast,hashlib,json,time
from pathlib import Path
import numpy as np
from research.v4.candidates.r3_06_actual_side55_tail_midpoint_v2_20260928 import root_tail as proven
from research.v4.candidates.r3_06_oven_only_side55_prefix_20260928.scene import OvenOnlyTrayScene
from .targets import derive
H=Path(__file__).resolve().parent
records=[]
class RecordedScene(OvenOnlyTrayScene):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.actual_rows=[];self.initial_contacts=proven.parent.contact_rows(self);self.initial_nondevice=proven.parent.initial_nondevice(self);self.initial_qpos=self.backend.sim.data._data.qpos.tolist();records.append(self)
  if any(x['non_grasp_hard_stop']for x in self.initial_contacts):raise RuntimeError('Initial non-grasp hard contact >1mm')
 def step(self,action):
  result=super().step(action);d=self.backend.sim.data._data;contacts=proven.parent.contact_rows(self)
  self.actual_rows.append({'sequence':int(self.sequence),'time_s':float(d.time),'action':np.asarray(action).tolist(),'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),'held':bool(self.grasp_assist.held),'EEF':d.site_xpos[self.grasp_assist._center].tolist(),'contacts':contacts,'original_risk':proven.parent.snapshot(self)})
  if any(x['non_grasp_hard_stop']for x in contacts):raise RuntimeError('Actual non-grasp hard contact >1mm; firstfailure')
  return result

def compiled(Scene=RecordedScene):
 f,src,prefixsha=proven.compiled_tail(Scene)
 seam='            _tail_prefix_identity(scene, details)'
 if src.count(seam)!=1:raise ValueError('Own seeded target seam missing/duplicate')
 src=src.replace(seam,'            _tail_plan = _derive_own_tail(scene, details)')
 ns=dict(f.__globals__);ns['_derive_own_tail']=derive
 exec(compile(src,str(H/'scripts.py')+'::original-branches-own-actual-midpoint','exec'),ns)
 return ns['run_case'],src,prefixsha

def run_case(seed,case):
 n=len(records);pools=[proven.parent.own.records,proven.parent.frame.records,proven.parent.transport.records];sizes=[len(x)for x in pools]
 f,src,sha=compiled()
 try:
  result=f(seed,case);scene=records[n]
  result.update(instance='R3-06',instruction=proven.parent.transport.INSTRUCTION,kind='CPU original analytic qualification, not model',actual_trace=scene.actual_rows,initial_contacts=scene.initial_contacts,initial_nondevice=scene.initial_nondevice,initial_qpos=scene.initial_qpos,source_original_prefix_sha256=sha,actual_generated_source_sha256=hashlib.sha256(src.encode()).hexdigest(),actual_contacts_hard_stop=any(c['non_grasp_hard_stop']for row in scene.actual_rows for c in row['contacts']),own_source_jitter_seed=int(seed),camera_visual_native_pending=True)
  if result['actual_contacts_hard_stop']:result['passed']=False
  return result
 finally:
  del records[n:]
  for pool,size in zip(pools,sizes):del pool[size:]
  proven.parent.INITIAL_METADATA.clear()
