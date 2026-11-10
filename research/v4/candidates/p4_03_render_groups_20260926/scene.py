"""Eight intentional static appearances only; no solver or general collider display."""
import numpy as np,mujoco
from research.v4.candidates.p4_03_gluegun_20260926.scene import GlueGunScene
APPEARANCE_NAMES=('tray_ceramic_collision','craft_work_mat','paper_red','paper_yellow','paper_blue','gun_stand_base','iron_stand_left_rail_collision','iron_stand_right_rail_collision')
def arrays(obj,skip=()):
 out={}
 for name in dir(obj):
  if name.startswith('_') or name in skip:continue
  value=getattr(obj,name)
  if isinstance(value,np.ndarray):out[name]=value.copy()
 return out

def same_array(a,b):
 # Treat corresponding NaN values as equal; shape/dtype and bytes otherwise checked.
 return a.shape==b.shape and a.dtype==b.dtype and bool(np.array_equal(a,b,equal_nan=True))
def integration(m,d):
 spec=mujoco.mjtState.mjSTATE_INTEGRATION
 value=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,value,spec)
 return value

def contact_snapshot(m,d):
 rows=[]
 for i,c in enumerate(d.contact[:d.ncon]):
  f=np.empty(6);mujoco.mj_contactForce(m,d,i,f)
  rows.append((int(c.geom1),int(c.geom2),float(c.dist),c.pos.copy(),c.frame.copy(),f))
 return rows

class VisibleGlueGunScene(GlueGunScene):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  try:
   m=self.backend.sim.model._model;d=self.backend.sim.data._data
   inactive=set()
   if d.nisland==0:inactive.update({'dof_island','dof_islandind','efc_island','island_dofind','island_efcind'})
   if not mujoco.mj_isSparse(m):inactive.update({'efc_AR_rowadr','efc_AR_rownnz'})
   before=arrays(m);state=arrays(d,skip=inactive);changes=[]
   integration_before=integration(m,d);time_before=float(d.time);contacts_before=contact_snapshot(m,d)
   inactive_shapes={n:list(getattr(d,n).shape) for n in sorted(inactive)}
   if self.backend.render_collision_mesh or not self.backend.render_visual_mesh:raise RuntimeError('normal visual rendering flags required')
   # These exactly sized/color-authored static surfaces were designed to be seen.
   # Do not expose group0 gun proxies or toggle renderer collision flags.
   for name in APPEARANCE_NAMES:
    gid=m.geom(name).id
    if m.geom_group[gid]!=0 or m.geom_rgba[gid,3]!=1:raise RuntimeError(f'appearance precondition: {name}')
    changes.append({'geom':name,'before_group':int(m.geom_group[gid]),'after_group':1,'rgba':m.geom_rgba[gid].tolist(),'size':m.geom_size[gid].tolist()});m.geom_group[gid]=1
   after=arrays(m);afterstate=arrays(d,skip=inactive)
   changed=[k for k in before if not same_array(before[k],after[k])]
   statechanged=[k for k in state if not same_array(state[k],afterstate[k])]
   integration_same=same_array(integration_before,integration(m,d));time_same=time_before==float(d.time);contacts_after=contact_snapshot(m,d)
   contacts_same=len(contacts_before)==len(contacts_after) and all(x[:3]==y[:3] and all(same_array(a,b) for a,b in zip(x[3:],y[3:])) for x,y in zip(contacts_before,contacts_after))
   if not integration_same or not time_same or not contacts_same:raise RuntimeError('authoritative integration/contact state changed')
   if changed!=['geom_group'] or statechanged:raise RuntimeError(f'nonrender mutation: {changed}/{statechanged}')
   diff=np.where(before['geom_group']!=after['geom_group'])[0]
   if set(diff)!=set(m.geom(n).id for n in APPEARANCE_NAMES):raise RuntimeError('unscoped group change')
   self.render_equivalence={'mjSTATE_INTEGRATION_size':int(integration_before.size),'mjSTATE_INTEGRATION_nan_equal':integration_same,'time_identical':time_same,'active_contacts_and_forces_identical':contacts_same,'contact_count':len(contacts_before),'inactive_solver_workspace_view_shapes':inactive_shapes,'inactive_solver_workspace_views_excluded':sorted(inactive),'nisland':int(d.nisland),'solver_sparse':bool(mujoco.mj_isSparse(m)),'model_arrays_compared':len(before),'data_arrays_compared':len(state),'changed_model_arrays':changed,'changed_data_arrays':statechanged,'physical_control_arrays_nan_equal':True,'specific_changes':changes,'render_collision_mesh':self.backend.render_collision_mesh,'render_visual_mesh':self.backend.render_visual_mesh}
   if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
  except BaseException:self.close();raise
