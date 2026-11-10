"""S2-02 own exact instructions/tea, current common release-props template."""
import json
from pathlib import Path
from research.v4.candidates.s2_01_seeded_yaw_20260927.scene import Scene as Parent
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'config.json').read_text())
INSTRUCTIONS=tuple(CONFIG['instructions'])
class Scene(Parent):
    protocol='S202-painkiller-grandpa-tea+common-release-props-yaw190-v1'
    instructions=INSTRUCTIONS
    def __init__(self,*args,**options):
        super().__init__(*args,**options);self.instance_id='S2-02';self.instructions=INSTRUCTIONS
        m=self.env.sim.model;geom=m.geom_name2id('milk_mug_1_milk_surface');mat=int(m.geom_matid[geom])
        if mat<0:raise ValueError('S202 original drink-surface material missing')
        m.mat_rgba[mat]=CONFIG['tea_rgba']
        self.tea_material_initialization={'rgba':list(CONFIG['tea_rgba']),'only_visual_material_changed':True,
                                          'new_forward_or_control_steps':0}
        # Final tea must be in the first actual public image, not cached milk.
        if self.camera_observations:self.raw=self.env.env._get_observations(force_update=True)
    def start_round(self,index):
        super().start_round(index);return self.instructions[index-1]
