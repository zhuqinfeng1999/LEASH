"""S2-05 exact Grandma/blood thinner/cocoa; runtime material only."""
from research.v4.candidates.s2_template_floor_completion_20260926.scene import Scene as OriginalScene

def apply_cocoa_display(scene):
    m=scene.env.sim.model;g=m.geom_name2id('milk_mug_1_milk_surface');material=int(m.geom_matid[g])
    if material<0:raise ValueError('original drink surface material missing')
    m.mat_rgba[material]=[.27,.105,.045,1.];m.mat_specular[material]=.3;m.mat_shininess[material]=.22
    # No additional forward, observations, controls or hidden dose decoration.
class Scene(OriginalScene):
    def __init__(self,*args,instance='S2-05',**kwargs):
        if instance!='S2-05':raise ValueError(instance)
        super().__init__(*args,instance=instance,**kwargs);apply_cocoa_display(self)
