"""Accepted E2 cat event with the existing public reader and v4.3 boundary."""
from types import FunctionType, SimpleNamespace

from research.v4.bridge.t1_lite import public_capture, runner as main_runner
from research.v4.bridge.swept_lite import Consumer, public_factory
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import bindings as acquisition
from research.v4.candidates.e2_cat_corrected_hold_20260926.facade import E2CorrectedHoldScene
from research.v4.candidates.e2_native_event_20260926.evaluator import E2Evaluator
from research.v4.candidates.e2_cat_v2_entry_20260926.name_binding import PROMPTS
from research.v4.candidates.e2_cat_v2_entry_20260926.public_workspace import locate_vla_target


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    if instance != 'E2-01':
        raise ValueError('This binding uses the accepted base cat instance')
    class Scene(E2CorrectedHoldScene):
        capture_agent = public_capture
    sweep = type('DictionaryPayloadSweep', (acquisition.ObservedPayloadSweep,), {
        '__init__': FunctionType(acquisition.ObservedPayloadSweep.__init__.__code__,
            {**acquisition.__dict__, 'SOURCE': 'dictionary'}, '__init__')})
    configured = SimpleNamespace(Scene=Scene, Evaluator=E2Evaluator,
        binding=SimpleNamespace(source='dictionary'),
        PublicWorkspaceFactory=public_factory(PROMPTS, locate_vla_target),
        bindings=SimpleNamespace(ObservedPayloadSweep=sweep))
    module = main_runner(instance, scene=Scene, evaluator=E2Evaluator,
        source='dictionary', prompts=PROMPTS, aliases={}, configured=configured,
        consumer_class=Consumer,
        runtime_overrides=runtime_overrides, tree_transform=tree_transform)
    module.lite_source_files = ['research/v4/bridge/t1_lite.py',
        'research/v4/bridge/swept_lite.py', 'research/v4/bridge/e2_lite.py']
    return module
