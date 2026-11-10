"""Final public reset/camera scene with only the consumption event replaced."""
from research.v4.candidates.s2_public_reset_20260926.higher_scene import HigherPoseScene
from .private_dynamics import DissolutionSafeDynamics
class DissolutionSafeScene(HigherPoseScene):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Initialization is already settled; this replacement changes no XML,
        # controller, pose, camera, pill size, instruction or scoring predicate.
        if self.private_dynamics.dose_count or self.private_dynamics.events:raise RuntimeError('cannot replace active S2 event history')
        self.private_dynamics=DissolutionSafeDynamics();self.private_dynamics.arm(self)
