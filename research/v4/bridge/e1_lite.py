"""Own E1 configurations over the existing E101 business callbacks."""
from types import SimpleNamespace

from research.v4.bridge.t1_lite import public_capture, runner as main_runner
from research.v4.bridge.swept_lite import public_factory, provider
from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import components
from research.v4.candidates.e1_future_public_binding_20260927.public_names import perception_queries
from research.v4.candidates.e101_v43_consumer_runtime_20260929.runtime import E101BaseConsumer
from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle


class E1BaseConsumer(E101BaseConsumer):
    def __init__(self, *args, **kwargs):
        # Preserve E101's public phase/geometry callbacks. The old constructor
        # selects only E1-01; this configuration supplies its own typed reader.
        common.Consumer.__init__(self, *args, provider_factory=provider, **kwargs)


class Consumer(TaskConstraints, OwnLeaseLifecycle, E1BaseConsumer):
    pass


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    original = components(instance)
    class Scene(original.Scene):
        capture_agent = public_capture
    prompts = perception_queries(instance)
    configured = SimpleNamespace(**{**vars(original), 'Scene':Scene,
        'PublicWorkspaceFactory':public_factory(prompts,
            original.PublicWorkspaceFactory.locate_vla_target)})
    module = main_runner(instance, scene=Scene, evaluator=original.Evaluator,
        source=original.binding.source, prompts=prompts, aliases={},
        configured=configured, consumer_class=Consumer,
        runtime_overrides=runtime_overrides, tree_transform=tree_transform)
    module.lite_source_files = ['research/v4/bridge/t1_lite.py',
        'research/v4/bridge/swept_lite.py', 'research/v4/bridge/e1_lite.py']
    return module
