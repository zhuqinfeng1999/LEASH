"""Select the actual accepted basket/candle/tag revision, without new physics."""
from dataclasses import replace
from types import SimpleNamespace
from research.v4.candidates.i_own_plain_config_20260927 import config as parent
from research.v4.candidates.user_rework_support_20260927.i_family import scene as family
from research.v4.candidates.user_rework_support_20260927.i_candle_front_v2 import scene as candle
from research.v4.candidates.i4_common_park_readable_tag_20260927 import scene as tags

IDS = tuple(f'I{f}-{v:02d}' for f in range(1, 5) for v in range(1, 4))
BINDINGS = tuple(parent.Binding(instance, 'ketchup', 'basket',
    ('knife', 'candle', 'metal can')[int(instance[-2:]) - 1],
    parent.INSTRUCTION, selected_module)
    for instance in IDS
    for selected_module in ((tags.__name__ if instance.startswith('I4-') else
                             candle.__name__ if instance.endswith('-02') else family.__name__),))

def binding(instance):
    try:
        return next(b for b in BINDINGS if b.instance == instance)
    except StopIteration:
        raise ValueError('Only the twelve registered own I instances') from None

def components(instance):
    b = binding(instance)
    module = tags if instance.startswith('I4-') else candle if instance.endswith('-02') else family
    create = module.create
    class Scene(parent.PublicFacade):
        instruction = b.instruction
        revision_instance = instance
        accepted_create = staticmethod(create)
        def __init__(self, seed=0, *, camera_observations=False, render_gpu_device_id=-1):
            self.backend, self.grasp_assist = create(instance, int(seed),
                camera_observations=camera_observations, render_gpu_device_id=render_gpu_device_id)
            self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
            self.raw = self.backend._get_observations(force_update=True)
            self.sequence = 0
            self.assist_events = []
    class Evaluator(parent.StrictEvaluator):
        def __init__(self):
            super().__init__(instance)
    return b, Scene, Evaluator
