"""Use the existing accepted R1 electronics/liquid scenes in the common runner."""
from importlib import import_module

from research.v4.bridge.t1_lite import runner as main_runner, public_capture
from research.v4.candidates.r1_02_t1_derived_20260926.offline_evaluator import Evaluator

# Current accepted carriers, including the qualified physical gamepad tail.
SCENES = {
    'R1-02': ('r1_02_t1_derived_20260926', 'T1R1CameraScene', 'compact camera', 'coffee mug'),
    'R1-03': ('r1_03_t1_derived_20260926', 'T1R1RemoteScene', 'TV remote', 'soup bowl'),
    'R1-04': ('r1_04_t1_derived_20260926', 'T1R1DryerScene', 'hairdryer', 'water basin'),
    'R1-05': ('r1_05_tail_qualification_20260926', 'Scene', 'game controller', 'juice pitcher'),
    'R1-06': ('r1_06_t1_derived_20260926', 'T1R1EreaderScene', 'ebook reader', 'milk pot'),
}


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    package, name, source, target = SCENES[instance]
    original = getattr(import_module('research.v4.candidates.' + package + '.scene'), name)
    class Scene(original):
        capture_agent = public_capture
    if instance == 'R1-05':
        from research.v4.candidates.r1_05_own_method_dispatch_20260927 import public_boundary as public
        prompts = public.NOUNS
        aliases = {**{x.lower(): 'item' for x in public.SOURCE_ALIASES},
                   **{x.lower(): 'box' for x in public.TARGET_ALIASES}}
    else:
        from research.v4.candidates.r1_derived_method_20260926 import public_boundary as public
        prompts, aliases = public.NOUNS[instance], public.aliases(instance)
    module = main_runner(instance, scene=Scene, evaluator=Evaluator, source='item',
        prompts=prompts, aliases=aliases,
        runtime_overrides=runtime_overrides, tree_transform=tree_transform)
    module.lite_source_files = ['research/v4/bridge/r1_family_lite.py',
        'research/v4/bridge/t1_lite.py', 'research/v4/bridge/full_corner_lite.py']
    return module
