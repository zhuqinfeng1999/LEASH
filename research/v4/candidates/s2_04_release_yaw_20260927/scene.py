"""S2-04 exact Tom/iron/juice display over the shared source190 physical template."""
import json
from pathlib import Path
from research.v4.candidates.s2_01_seeded_yaw_20260927.scene import Scene as Parent
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'config.json').read_text())
INSTRUCTIONS=tuple(CONFIG['instructions'])
class Scene(Parent):
    protocol='S204-iron-Tom-juice+common-release-props-yaw190-v1'
    instructions=INSTRUCTIONS
    def __init__(self,*args,**options):
        super().__init__(*args,**options);self.instance_id='S2-04';self.instructions=INSTRUCTIONS
        m=self.env.sim.model;geom=m.geom_name2id('milk_mug_1_milk_surface');mat=int(m.geom_matid[geom])
        if mat<0:raise ValueError('S204 original drink-surface material missing')
        m.mat_rgba[mat]=CONFIG['juice_rgba'];m.mat_specular[mat]=CONFIG['juice_specular'];m.mat_shininess[mat]=CONFIG['juice_shininess']
        self.juice_material_initialization={'rgba':list(CONFIG['juice_rgba']),'specular':CONFIG['juice_specular'],
                                          'shininess':CONFIG['juice_shininess'],'only_visual_material_changed':True,
                                          'new_forward_or_control_steps':0}
        # Actual first public frame uses the final registered juice material.
        if self.camera_observations:self.raw=self.env.env._get_observations(force_update=True)
    def start_round(self,index):
        super().start_round(index);return self.instructions[index-1]
