"""Offline S2-01 completion repair; exact original final carrier/dose events."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.s2_dissolution_boundary_20260926.scene import DissolutionSafeScene
from research.v4.candidates.s2_scene_20260924.scene import INSTRUCTIONS
from research.v4.candidates.s2_scene_20260924.private_dynamics import PILLS,MUG,crossed_milk
from research.v4.candidates.s2_template_floor_completion_20260926 import probe as common
from research.v4.candidates.s2_template_floor_completion_20260926.evaluator import Evaluator
from research.v4.candidates.s2_template_floor_completion_20260926.support import basket_evidence,PhysicalBasketTracker

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTANCE='S2-01';EXPECTED=common.EXPECTED

class AuditedScene(DissolutionSafeScene):
    """No extra forward/observations or display overlay at initialization."""
    instructions=INSTRUCTIONS
    def __init__(self,*args,instance=INSTANCE,visual_dir=None,**kwargs):
        if instance!=INSTANCE:raise ValueError(instance)
        self.instance_id=INSTANCE
        super().__init__(*args,**kwargs)
        self.audit=[];self.initial_mjcf=self.env.env.model.get_xml()
        self.visual_dir=Path(visual_dir)if visual_dir else None
        if self.visual_dir:self.visual_dir.mkdir(exist_ok=False,parents=True);self.save_views('initial')
        # Same private event audit as the qualified S202 common probe.
        original_event=self.private_dynamics.on_step
        def audited(scene):
            sim=scene.env.sim;objects=scene.env.env.objects_dict
            xyz=lambda n:np.asarray(sim.data.get_joint_qpos(objects[n].joints[0])[:3],float).copy()
            mug=xyz(MUG);before=scene.private_dynamics.dose_count;seen=set(scene.private_dynamics.dissolved_pills)
            entries={n:crossed_milk(mug,scene.private_dynamics.previous_positions.get(n,xyz(n)),xyz(n))for n in PILLS if n not in seen}
            eqs={n:bool(sim.data.eq_active[sim.model._model.equality('v42_s2_'+n+'_weld').id])for n in (*PILLS,'ketchup_1')}
            events=original_event(scene)
            self.audit.append(dict(sim_time_s=float(sim.data.time),round=scene.round_index,executed_controller_step=scene.sequence,
                dose_before=before,dose_after=scene.private_dynamics.dose_count,active_welds=eqs,
                physical_milk_entries=entries,events=events,dissolved_after=sorted(scene.private_dynamics.dissolved_pills),
                dissolved_positions_after={n:xyz(n).tolist()for n in scene.private_dynamics.dissolved_pills}))
            return events
        self.private_dynamics.on_step=audited
    save_views=common.AuditedScene.save_views
    def close(self):
        if self.visual_dir:self.save_views('final');(self.visual_dir/'scene.xml').write_text(self.initial_mjcf)
        super().close()

def run(phase,seed,**kwargs):
    # Reuses exact original controls and the already qualified strict AST
    # basket-observe replacement/evaluator; no new executor or physics.
    with patch.object(common,'AuditedScene',AuditedScene):result=common.run(INSTANCE,phase,seed,**kwargs)
    result.update(offline_predicate_version='S201-original-carrier+common-S202-full-footprint-floor-v1',
                  instruction_changed=False,physics_changed=False,old_labels_rescored=False)
    return result
