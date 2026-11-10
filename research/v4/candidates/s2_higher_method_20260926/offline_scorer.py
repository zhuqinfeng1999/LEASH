"""Read-only logging facade over S2's existing per-low-level-step evaluator.

Scene.step already observes the evaluator after every real control. The callback
must not observe again at that timestamp (round2 support needs increasing time).
"""
from research.v4.candidates.s2_scene_20260924.evaluator import Evaluator,snapshot
class S2ControlScorer:
    def __init__(self,scene):
        self.scene=scene;self.evaluator=Evaluator();scene.attach_evaluator(self.evaluator)
        self.round_number=0;self.steps_seen=0;self.round_boundaries=[];self.terminal_provider=None
    def begin_round(self,number):
        if number!=self.round_number+1 or number not in (1,2,3):raise ValueError('S2 round order changed')
        self.evaluator.begin_round(self.scene,number);self.round_number=number;self.round_boundaries.append(number)
    def observe(self,scene):
        if scene is not self.scene or scene.round_number!=self.round_number:raise ValueError('S2 scorer scene/round changed')
        self.steps_seen=int(scene.sequence)
        return {'event':'physical_state_offline_only',**snapshot(scene),'actual_controller_steps':self.steps_seen}
    def finish(self):
        terminal=self.terminal_provider() if self.terminal_provider else None
        raw=self.evaluator.finish(terminal.get('status') if terminal else None)
        raw.update(round_boundaries=list(self.round_boundaries),actual_controller_steps=self.steps_seen)
        return raw
