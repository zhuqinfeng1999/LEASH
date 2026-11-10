"""Same actual3mm floor; one explicit local contact calibration, all methods."""
import hashlib
import numpy as np
from research.v4.candidates.p1_04_hover_release_20260927.scripts import RecordedScene

def model_arrays(m):
 result={}
 for name in dir(m):
  try:value=getattr(m,name)
  except Exception:continue
  if isinstance(value,np.ndarray):result[name]=hashlib.sha256(value.tobytes()).hexdigest()
 return result

class Scene(RecordedScene):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  m=self.backend.sim.model._model
  target=self.backend.objects_dict[self.target_name]
  name=next(n for n in target.contact_geoms if n.endswith('_plate'))
  gid=m.geom(name).id;before=model_arrays(m);old=dict(priority=int(m.geom_priority[gid]),solref=m.geom_solref[gid].tolist(),solimp=m.geom_solimp[gid].tolist())
  before_options=(m.opt.timestep,m.opt.enableflags,m.opt.disableflags)
  m.geom_priority[gid]=1;m.geom_solref[gid]=[.004,1.];m.geom_solimp[gid,:3]=[.99,.99,.001]
  after=model_arrays(m);changed=[n for n in before if before[n]!=after[n]]
  if set(changed)!={'geom_priority','geom_solref','geom_solimp'}or before_options!=(m.opt.timestep,m.opt.enableflags,m.opt.disableflags):
   self.close();raise RuntimeError('Only three local floor contact arrays may change')
  self.floor_calibration=dict(geom=name,geom_id=int(gid),old=old,new=dict(priority=int(m.geom_priority[gid]),solref=m.geom_solref[gid].tolist(),solimp=m.geom_solimp[gid].tolist()),changed_arrays=changed,all_other_arrays_exact=True,timestep_s=float(m.opt.timestep),enableflags=int(m.opt.enableflags),disableflags=int(m.opt.disableflags),unchanged_actual_size=m.geom_size[gid].tolist(),unchanged_actual_pos=m.geom_pos[gid].tolist(),arrays_before=before,arrays_after=after)
