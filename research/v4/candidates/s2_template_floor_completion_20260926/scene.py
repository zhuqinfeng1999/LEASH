"""Configuration-only S2 family wrapper over the final same-world carrier."""
import json
from pathlib import Path
from research.v4.candidates.s2_dissolution_boundary_20260926.scene import DissolutionSafeScene
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'config.json').read_text())
class Scene(DissolutionSafeScene):
    def __init__(self,*args,instance='S2-02',**kwargs):
        if instance not in CONFIG:raise ValueError(instance)
        self.instance_id=instance;self.instructions=tuple(CONFIG[instance]['instructions'])
        super().__init__(*args,**kwargs)
        if instance=='S2-02':
            m=self.env.sim.model;g=m.geom_name2id('milk_mug_1_milk_surface');material=int(m.geom_matid[g])
            if material<0:raise ValueError('original drink surface material missing')
            m.mat_rgba[material]=[.46,.23,.055,1.]
        self.env.sim.forward()
        self.raw=self.env.env._get_observations(force_update=True)
    def start_round(self,round_index):
        super().start_round(round_index)
        return self.instructions[round_index-1]
