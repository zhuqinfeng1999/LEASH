"""Water material writes only; retain exact parent forward/observer cadence."""
from research.v4.candidates.s2_template_floor_completion_20260926.scene import Scene as OriginalScene

def apply_water_display(scene):
    m=scene.env.sim.model;g=m.geom_name2id('milk_mug_1_milk_surface');material=int(m.geom_matid[g])
    if material<0:raise ValueError('original liquid surface material missing')
    m.mat_rgba[material]=[.32,.48,.57,.55];m.mat_specular[material]=.5;m.mat_shininess[material]=.6
    # Do not forward or poll an extra observer: parent already performed those.
    # The normal public capture renders fresh observations after this tint.
class Scene(OriginalScene):
    def __init__(self,*args,instance='S2-03',**kwargs):
        if instance!='S2-03':raise ValueError(instance)
        super().__init__(*args,instance=instance,**kwargs);apply_water_display(self)
