"""Own S205 blood-thinner/Grandma/cocoa settings over common190 CC0 physical props."""
import os
os.environ['LIBERO_CONFIG_PATH']='__LEASH_ROOT__/configs/runtime/libero'
import json
from pathlib import Path
from research.v4.candidates.s2_01_seeded_yaw_20260927.scene import Scene as Parent
from research.v4.candidates.s2_05_cocoa_config_20260926.scene import apply_cocoa_display
HERE=Path(__file__).resolve().parent;CONFIG=json.loads((HERE/'config.json').read_text());INSTRUCTIONS=tuple(CONFIG['instructions'])
class Scene(Parent):
 protocol='S205-blood-thinner-Grandma-cocoa+common-release-props-yaw190-v1'
 instructions=INSTRUCTIONS
 def __init__(self,*args,**options):
  super().__init__(*args,**options);self.instance_id='S2-05';self.instructions=INSTRUCTIONS;apply_cocoa_display(self)
  m=self.env.sim.model;g=m.geom_name2id('milk_mug_1_milk_surface');mat=int(m.geom_matid[g])
  self.cocoa_material_initialization=dict(rgba=m.mat_rgba[mat].tolist(),specular=float(m.mat_specular[mat]),shininess=float(m.mat_shininess[mat]),only_display_material_changed=True,added_physical_controls=0,added_forward=0,final_material_refreshes=0)
  # GPU initial packet must be from final cocoa, not parent cached milk.
  if self.camera_observations:
   self.raw=self.env.env._get_observations(force_update=True);self.cocoa_material_initialization['final_material_refreshes']=1
 def start_round(self,index):
  super().start_round(index);return self.instructions[index-1]
