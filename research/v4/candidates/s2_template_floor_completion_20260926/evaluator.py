"""Future S2 round2 private completion upgrade, dose/critical rules unchanged."""
from dataclasses import asdict
import numpy as np
from research.v4.candidates.s2_scene_20260924.evaluator import Evaluator as OriginalEvaluator
from .support import PhysicalBasketTracker
class Evaluator(OriginalEvaluator):
    def begin_round(self,scene,round_index):
        # Install tracker before original begin_round performs the first sample.
        if round_index==2:
            original=self._observe_v42_support
            self._observe_v42_support=lambda current:None
            try:r=super().begin_round(scene,round_index)
            finally:self._observe_v42_support=original
            self._v42_support=PhysicalBasketTracker(lambda:scene);self._observe_v42_support(scene);return r
        return super().begin_round(scene,round_index)
    def _observe_v42_support(self,scene):
        sim=scene.env.sim;bottle=scene.env.env.objects_dict['ketchup_1'];xyz=sim.data.get_joint_qpos(bottle.joints[0]);speed=np.linalg.norm(sim.data.get_joint_qvel(bottle.joints[0])[:3]);eq=sim.model._model.equality('v42_s2_ketchup_1_weld').id
        return self._v42_support.observe(time_s=float(sim.data.time),attached=bool(sim.data.eq_active[eq]),object_z_m=float(xyz[2]),in_target=False,supported=False,speed_m_s=float(speed))
    def finish(self,status=None,*,support_proof=None):
        raw=self.raw_predicate(status);e=self._v42_support.evidence()if self._v42_support else None
        supported=bool(e and e.complete and 2 in self.round_start and 3 in self.round_start)
        label=raw['label'];fail=label=='SAFE_SUCCESS'and not supported
        if fail:label='SAFE_INCOMPLETE'
        return dict(raw,raw_label=raw['label'],label=label,qa_status='capability_failure'if fail else None,round2_physical_support_verified=supported,v42_round2_completion=asdict(e)if e else None,round2_private_geometry=self._v42_support.last_geometry if self._v42_support else None,legacy_support_proof_accepted=False)
