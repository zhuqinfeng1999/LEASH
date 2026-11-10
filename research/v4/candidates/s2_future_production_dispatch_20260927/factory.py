"""Candidate future S2 dispatch only; historical runner and physics are untouched."""
from dataclasses import dataclass
from types import MappingProxyType
from unittest.mock import patch
from research.v4.candidates.s2_higher_method_20260926.rpent_facade import S2RPentFacade as OriginalFacade
from research.v4.candidates.s2_higher_method_20260926 import method_core as original_core
from research.v4.candidates.s2_01_strict_method_scorer_20260927.scorer import S2StrictControlScorer
from research.v4.candidates.s2_01_seeded_yaw_20260927.scene import Scene as S201Scene
from research.v4.candidates.s2_02_release_yaw_20260927.scene import Scene as S202Scene
from research.v4.candidates.s2_03_release_yaw_20260927.scene import Scene as S203Scene
from research.v4.candidates.s2_04_release_yaw_20260927.scene import Scene as S204Scene
from research.v4.candidates.s2_05_release_yaw_20260927.scene import Scene as S205Scene


class ProductionFacade(OriginalFacade):
    """Accepted factory selected by a module-level class, not caller truth data.

    The original public capture implementation and two640x480/calibration startup
    are reused. Its diagnostic injection hook is used internally only to select
    this audited production class; callers cannot supply a custom scene factory.
    """
    scene_class = None
    instance_id = None

    def __init__(self, seed=0, *, camera_observations=False, render_gpu_device_id=0):
        if self.scene_class is None:
            raise TypeError('Select an exact registered S2 production facade')
        super().__init__(seed, camera_observations=camera_observations,
                         render_gpu_device_id=render_gpu_device_id,
                         diagnostic_scene_factory=self.scene_class)

    @property
    def instructions(self):
        return tuple(self.scene_class.instructions)

    @property
    def instruction(self):
        return self.instructions[self.round_number-1]


class S201Facade(ProductionFacade):
    scene_class = S201Scene
    instance_id = 'S2-01'

class S202Facade(ProductionFacade):
    scene_class = S202Scene
    instance_id = 'S2-02'

class S203Facade(ProductionFacade):
    scene_class = S203Scene
    instance_id = 'S2-03'

class S204Facade(ProductionFacade):
    scene_class = S204Scene
    instance_id = 'S2-04'

class S205Facade(ProductionFacade):
    scene_class = S205Scene
    instance_id = 'S2-05'


@dataclass(frozen=True)
class Binding:
    instance_id: str
    facade_class: type
    scorer_class: type
    instructions: tuple[str, ...]


BINDINGS = MappingProxyType({cls.instance_id: Binding(cls.instance_id,cls,
    S2StrictControlScorer,tuple(cls.scene_class.instructions))
    for cls in (S201Facade,S202Facade,S203Facade,S204Facade,S205Facade)})


def binding(instance_id):
    try:return BINDINGS[instance_id]
    except (KeyError,TypeError):raise ValueError('Unregistered S2 instance: '+repr(instance_id))from None


def run_core(instance_id, **kwargs):
    """Scoped own instruction binding; original persistent-session core remains.

    This is a future integration hook, not a new CLI or authorization to run a
    model. One owner-thread/process per episode, as in the existing core.
    """
    spec=binding(instance_id)
    if not isinstance(kwargs.get('scene'),spec.facade_class):
        raise ValueError('S2 core requires its exact registered production facade')
    with patch.object(original_core,'INSTRUCTIONS',spec.instructions):
        return original_core.run_s2_core(**kwargs)
