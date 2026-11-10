"""Existing derived E2 scenes with their own public object vocabulary."""
import importlib
from types import FunctionType, SimpleNamespace

from research.v4.bridge.t1_lite import public_capture, runner as main_runner
from research.v4.bridge.swept_lite import Consumer, public_factory
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import bindings as acquisition
from research.v4.candidates.e2_native_event_20260926.evaluator import E2Evaluator
from research.v4.candidates.e2_laundry_method_20260926.public_intent import infer_public_goal


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    if instance == 'E2-02':
        from research.v4.candidates.e2_laundry_cat_20260926.facade import E2LaundryScene as Original
        from research.v4.candidates.e2_laundry_method_20260926.name_binding import PROMPTS
        source = 'dumbbell'
        prompts = PROMPTS
        source_aliases = ('dumbbell','hand weight','weight')
        target_aliases = ('laundry basket','basket','laundry bin','container','box')
        target = 'laundry basket'
    else:
        from research.v4.candidates.e2_dog_own_method_dispatch_20260927.config import binding
        b = binding(instance)
        Original = getattr(importlib.import_module(b.scene_module), b.scene_class)
        source = b.source
        prompts = {b.source:b.source, b.target:b.target, 'dog':'dog'}
        source_aliases, target_aliases, target = b.source_aliases, b.target_aliases, b.target
    def locate(prompt, public):
        role = infer_public_goal(prompt, source_aliases=source_aliases,
            target_aliases=target_aliases,
            public_held=public.evidence.robot.hold_estimate == 'held')
        noun = source if role == 'source' else target if role == 'target' else None
        box = public.box(noun) if noun else None
        if box is None: return None
        return noun, tuple((lo+hi)/2 for lo,hi in zip(box.lower,box.upper))
    class Scene(Original):
        capture_agent = public_capture
    sweep = type('OwnE2PayloadSweep', (acquisition.ObservedPayloadSweep,), {
        '__init__':FunctionType(acquisition.ObservedPayloadSweep.__init__.__code__,
            {**acquisition.__dict__, 'SOURCE':source}, '__init__')})
    configured = SimpleNamespace(Scene=Scene, Evaluator=E2Evaluator,
        binding=SimpleNamespace(source=source),
        PublicWorkspaceFactory=public_factory(prompts, locate),
        bindings=SimpleNamespace(ObservedPayloadSweep=sweep))
    module = main_runner(instance, scene=Scene, evaluator=E2Evaluator,
        source=source, prompts=prompts, aliases={}, configured=configured,
        consumer_class=Consumer,
        runtime_overrides=runtime_overrides, tree_transform=tree_transform)
    module.lite_source_files = ['research/v4/bridge/t1_lite.py',
        'research/v4/bridge/swept_lite.py', 'research/v4/bridge/e2_family_lite.py']
    return module
