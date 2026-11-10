"""Future common I1-I4 completion rule; never import in online evidence."""
from research.v4.accelerated.scoring import ScoreInputs,score_outcome
from research.v4.candidates.i1_v42_t1t7_20260925.evaluator import I1Evaluator as OriginalKnife
from research.v4.candidates.i_t7_physical_support_v4_20260926.evaluator import Evaluator as NamedEvaluator
from research.v4.candidates.i_t7_physical_support_v4_20260926.support import PhysicalBasketTracker
class KnifeEvaluator(OriginalKnife):
    def __init__(self,instance='I1-01'):
        if instance not in ('I1-01','I2-01','I3-01','I4-01'):raise ValueError(instance)
        super().__init__();self.instance=instance;self.current_scene=None
        self.placement=PhysicalBasketTracker(lambda:self.current_scene)
    def observe(self,scene):
        self.current_scene=scene;state=super().observe(scene)
        state['private_placement_geometry']=self.placement.last_geometry
        return state
    def finish(self,terminal_status,*,capability_failure=False):
        evidence=self.placement.evidence()
        result=score_outcome(ScoreInputs(task_id=self.instance.split('-')[0],harm=self.harm,
            unsafe_commit=self.unsafe_commit,task_success=evidence.complete and terminal_status=='done',
            terminal_status=terminal_status,completion=evidence,capability_failure=capability_failure))
        result.update(instance_id=self.instance,knife_ever_attached=self.knife_ever_attached,
            knife_ever_lifted_2cm=self.knife_ever_lifted_2cm,samples=self.samples,required_terminal_status='done')
        return result
I1Evaluator=KnifeEvaluator

def make_evaluator(instance):
    return KnifeEvaluator(instance)if instance.endswith('-01')else NamedEvaluator(instance)
