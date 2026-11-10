"""Same actual screwdriver colliders made visible, matching qualified E1 asset."""
from research.v4.candidates.p2_03_screwdriver_handoff_20260926.scene import P2ScrewdriverFacade as Physical,INSTRUCTION,ROOT

def expose(scene):
 scene.visual_render_changes=[]
 for n in scene.backend.objects_dict[scene.item_name].contact_geoms:
  gid=scene.backend.sim.model.geom_name2id(n);old=int(scene.backend.sim.model.geom_group[gid]);scene.backend.sim.model.geom_group[gid]=1;scene.visual_render_changes.append(dict(geom=n,group_before=old,group_after=1,collision_unchanged=True))
 if scene.backend.use_camera_obs:scene.raw=scene.backend._get_observations(force_update=True)
class P2ScrewdriverFacade(Physical):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);expose(self)
