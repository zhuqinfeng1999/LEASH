"""Accepted P4 physical factories with their existing full-corner Consumer."""
from research.v4.bridge.t1_lite import runner as main_runner, public_capture


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    from research.v4.candidates.user_revision_real_p4_v4_20260928.config import own_components
    b, parent, evaluator = own_components(instance)
    class Scene(parent):
        capture_agent = public_capture
    prompts, aliases = b.vocabulary()
    module = main_runner(instance, scene=Scene, evaluator=evaluator, source=b.source,
                         prompts=prompts, aliases=aliases,
                         runtime_overrides=runtime_overrides, tree_transform=tree_transform)
    module.lite_source_files = ['research/v4/bridge/t1_lite.py', 'research/v4/bridge/p4_lite.py',
        'research/v4/bridge/full_corner_lite.py',
        'research/v4/candidates/p401_v43_actual_consumer_20260929/runtime.py']
    return module
