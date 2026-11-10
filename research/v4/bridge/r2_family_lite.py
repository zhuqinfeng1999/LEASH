"""Accepted R2 derived factories and public nouns in the common runtime."""
from unittest.mock import patch

from research.v4.bridge.t1_lite import public_capture, runner as main_runner


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    if instance == 'R2-02':
        from research.v4.candidates.r2_02_natural_method_20260926 import run_method as original
        from research.v4.candidates.r2_02_natural_method_20260926 import public_boundary as public
        Original, Evaluator = original.Scene, original.Evaluator
        prompts, aliases = public.PUBLIC_NOUNS, public.ALIASES
    elif instance == 'R2-03':
        from research.v4.candidates.r2_03_natural_method_20260926 import run_method as original
        from research.v4.candidates.r2_03_natural_method_20260926 import public_boundary as public
        Original, Evaluator = original.T1R2TealightScene, original.Evaluator
        prompts, aliases = public.PUBLIC_NOUNS, public.ALIASES
    else:
        from research.v4.candidates.r2_remaining_method_20260926 import run_method as original
        from research.v4.candidates.r2_remaining_method_20260926 import public_boundary as public
        Original, Evaluator = original.CONFIG[instance][:2]
        prompts, aliases = public.NOUNS[instance], public.ALIASES[instance]

    class Scene(Original):
        capture_agent = public_capture
        def __init__(self, *args, **kwargs):
            # Retain the approved Panda visual transform at the existing one
            # compilation. Retired multi-model diagnostic comparisons are not
            # invoked; no controller, geometry or task setting is changed.
            from robosuite.utils.binding_utils import MjSim
            from research.v4.candidates.panda_release_factory_optin_20260927.helper import source_identity
            compile_model = MjSim.from_xml_string
            with patch.object(MjSim, 'from_xml_string', classmethod(
                    lambda cls, xml:compile_model(source_identity(xml)))):
                super().__init__(*args, **kwargs)
        @property
        def backend(self):
            return self._env if hasattr(self, '_env') else self.env.env
        @backend.setter
        def backend(self, value):
            self._env = value

    module = main_runner(instance, scene=Scene, evaluator=Evaluator,
        source='item', prompts=prompts, aliases=aliases,
        runtime_overrides=runtime_overrides, tree_transform=tree_transform)
    module.lite_source_files = ['research/v4/bridge/t1_lite.py',
        'research/v4/bridge/full_corner_lite.py', 'research/v4/bridge/r2_family_lite.py']
    return module
