"""Private R5 hazard sampler plus existing strict score, never online evidence."""
from research.v4.candidates.r5_v42_label_large_20260925.evaluator import Evaluator as Original
from research.v4.accelerated.scoring import ScoreInputs,score_outcome

class Evaluator:
    def __init__(self):
        self.raw=None;self.ever_attached=False
    def observe(self,scene):
        if self.raw is None:self.raw=Original(scene)
        row=self.raw.observe(scene)
        self.ever_attached|=row['attached']
        return row
    def finish(self,status=None,*,judge=None,capability_failure=False,infrastructure_fault=None):
        if self.raw is None:raise ValueError('no actual physical evaluator sample')
        score=score_outcome(ScoreInputs(task_id='R5',
            harm=self.raw.first['HARM'] is not None,
            unsafe_commit=self.raw.first['UNSAFE_COMMIT'] is not None,
            terminal_status=status,target_ever_attached=self.ever_attached,
            refusal_judge=judge,capability_failure=capability_failure,
            infrastructure_fault=infrastructure_fault))
        return {**score,'first_hit_sim_time':dict(self.raw.first),
            'ever_attached':self.ever_attached,'sample_count':self.raw.samples,
            'last_truth':self.raw.last}
