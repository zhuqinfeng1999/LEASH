"""Accepted I-family scenes; the instruction comes only from the user channel."""
from types import FunctionType, SimpleNamespace

from research.v4.bridge.t1_lite import public_capture, runner as main_runner
from research.v4.bridge.swept_lite import Consumer, public_factory
from research.v4.candidates.i1_v42_t1t7_20260925.facade import I1Facade
from research.v4.candidates.i1_v42_t1t7_20260925.gate_macro_v2 import ALIASES, PROMPTS, I1PayloadEnvelope
from research.v4.candidates.i_common_floor_completion_20260926.evaluator import KnifeEvaluator, NamedEvaluator
from research.v4.candidates.r2_02_natural_method_20260926.public_boundary import PublicWorkspaceReader


def public_inputs(instance):
    """Reuse the instance's declared public nouns, without interpreting its sign."""
    if instance.endswith('-01'):
        prompts, aliases = PROMPTS, ALIASES
    else:
        from research.v4.candidates.i_user_revision_real_binding_20260928 import public_names
        prompts = public_names.perception_queries(instance)
        aliases = {name: noun for name, noun in ALIASES.items()
                   if noun in prompts or noun == 'person'}
        if instance.endswith('-02'):
            aliases.update({'candle': 'candle', 'lit candle': 'candle'})
        else:
            aliases.update({name: 'metal can' for name in
                            ('metal can', 'can', 'soup can', 'steel can')})
    original = PublicWorkspaceReader.locate_vla_target
    parser = FunctionType(original.__code__, {**original.__globals__, 'ALIASES': aliases},
                          original.__name__, original.__defaults__, original.__closure__)
    def locate(prompt, public):
        return parser(None, prompt, public)
    return prompts, aliases, locate


def accepted_components(instance):
    """Select the existing accepted layout and its own private evaluator."""
    if instance not in {f'I{family}-{number:02}' for family in range(1,5) for number in range(1,4)}:
        raise ValueError('This binding uses the twelve accepted I instances')
    if instance.startswith('I4-'):
        from research.v4.candidates.i4_common_park_readable_tag_20260927.scene import create
    elif instance.endswith('-02'):
        from research.v4.candidates.user_rework_support_20260927.i_candle_front_v2.scene import create
    else:
        from research.v4.candidates.user_rework_support_20260927.i_family.scene import create
    return create, KnifeEvaluator if instance.endswith('-01') else NamedEvaluator


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    create, evaluator_class = accepted_components(instance)
    prompts, aliases, locate = public_inputs(instance)
    class Scene(I1Facade):
        capture_agent = public_capture
        def __init__(self, seed=0, *, camera_observations=True, render_gpu_device_id=0):
            self.backend, self.grasp_assist = create(instance, seed,
                camera_observations=camera_observations,
                render_gpu_device_id=render_gpu_device_id)
            self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
            self.raw = self.backend._get_observations(force_update=True)
            self.sequence, self.assist_events = 0, []
    class Evaluator(evaluator_class):
        def __init__(self):
            super().__init__(instance)
    class PayloadEnvelope(I1PayloadEnvelope):
        def __init__(self, public):
            super().__init__()
    configured = SimpleNamespace(Scene=Scene, Evaluator=Evaluator,
        binding=SimpleNamespace(source='ketchup'),
        PublicWorkspaceFactory=public_factory(prompts, locate),
        bindings=SimpleNamespace(ObservedPayloadSweep=PayloadEnvelope))
    module = main_runner(instance, scene=Scene, evaluator=Evaluator,
        source='ketchup', prompts=prompts, aliases=aliases, configured=configured,
        consumer_class=Consumer,
        runtime_overrides=runtime_overrides, tree_transform=tree_transform)
    module.lite_source_files = ['research/v4/bridge/t1_lite.py',
        'research/v4/bridge/swept_lite.py', 'research/v4/bridge/i_lite.py']
    return module
