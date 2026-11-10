"""Accepted R3 physical factories and own public payloads, without custody."""
from types import FunctionType, ModuleType, SimpleNamespace
from unittest.mock import patch
import numpy as np

from research.v4.candidates.r3_family_method_config_20260927.config import binding, resolve
from research.v4.candidates.r3_family_method_config_20260927 import hooks
from research.v4.candidates.r301_v43_same_packet_public_provider_prepare_20260929 import provider
from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle
from research.v4.bridge.t1_lite import public_capture


def scene_type(instance):
    if instance in {'R3-03','R3-04','R3-06'}:
        from research.v4.candidates.r3_current_real_front_consumer_20260928.config import binding as current
        Original = resolve(current(instance).factory)
        class Scene(Original):
            capture_agent = public_capture
        return Scene
    if instance == 'R3-05':
        from research.v4.candidates.r305_panel_normal_tangent_current_native_hd_20260928 import selection as front
        class Scene(front.own.RobotFacingLugY80Scene):
            capture_agent = public_capture
            def __init__(self, *args, **kwargs):
                with patch.object(front.own.parent.ENV, '_load_model', front.front_loader):
                    front._INIT(self, *args, **kwargs)
        return Scene
    if instance == 'R3-02':
        from research.v4.candidates.r302_panel_normal_tangent_current_native_hd_20260928.scene import FrontFlatLedgeTall350Env
        from research.v4.candidates.r302_horizontal_short_v_20260928 import scene as horizontal
        from research.v4.candidates.r302_front_slot_short_v_20260928 import scene as front
        from research.v4.candidates.r302_short_v_raised_neutral_20260928.scene import CANONICAL_RAISED_PARK
        class Scene(horizontal.ShortVHorizontalCanScene):
            capture_agent = public_capture
            startup_control_count = 80
            def __init__(self, *args, **kwargs):
                # Same accepted reset configuration and fixed 80 OPEN controls.
                # The retired constructor's contact/model/AST reports are not
                # business actions and are not created in the run path.
                with patch.object(front.source_config, 'configuration', front.front_configuration), \
                     patch.object(front.source_config, 'NEUTRAL_QPOS', CANONICAL_RAISED_PARK), \
                     patch.object(horizontal, 'CentralShortVEnv', FrontFlatLedgeTall350Env):
                    super().__init__(*args, **kwargs)
                for _ in range(self.startup_control_count):
                    self.step(np.asarray([0.,0.,0.,0.,0.,0.,-1.]))
        return Scene
    raise ValueError('Derived R3 binding requires R3-02 through R3-06')


def runtime(instance):
    roles = binding(instance)
    functions = provider.prepared_functions()
    def sweep(public):
        return functions.observed_sweep(roles, public)
    def proposal(name, args, public, locate, held_sweep=None):
        return functions.proposal(roles, name, args, public, locate)
    components = SimpleNamespace(binding=roles, bindings=SimpleNamespace(ObservedPayloadSweep=sweep))
    def factory(_components, scene, adapter, sam_client):
        value = provider.SamePacketR301PublicProvider(scene, adapter, sam_client)
        value.roles = roles
        return value
    original = common.Consumer.resolve
    resolver = FunctionType(original.__code__, {**original.__globals__, 'proposal_for_p3':proposal},
        original.__name__, original.__defaults__, original.__closure__)
    class Base(common.Consumer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **dict(kwargs, components=components, provider_factory=factory))
        resolve = resolver
    consumer = type('OwnR3Consumer', (TaskConstraints, OwnLeaseLifecycle, Base), {})
    return SimpleNamespace(**{**vars(common), 'Consumer':consumer, 'components':components})


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    from research.v4.bridge.run_b0 import ROOT, lite_runtime, bind_lite_main, active_call_budgets
    from research.v4.candidates.r301_v43_actual_main_source_prepare_20260929 import binding as main_binding
    entry = hooks.entry_module(instance)
    entry.LowerAimR3Scene = scene_type(instance)
    path = ROOT/'research/v4/candidates/r3_v42_visibility_20260926/run_r3.py'
    tree, _ = main_binding.rewrite(path.read_text())
    module = ModuleType('research.v4.bridge._'+instance.replace('-','_'))
    module.__package__ = 'research.v4.candidates.r3_v42_visibility_20260926'
    module.__dict__.update(lite_runtime(instance))
    module.V43 = runtime(instance)
    module.V43.make_reviewer_adapter = module.lite_reviewer_adapter = lite_runtime(instance)['V43'].make_reviewer_adapter
    module.V43.structured_fault = main_binding.structured_fault
    module.__dict__.update(OWN_ENTRY=entry, OWN_HASHES=lambda:{})
    if runtime_overrides is not None:
        module.__dict__.update(runtime_overrides(vars(module)))
    if tree_transform is not None:
        tree = tree_transform(tree)
    exec(compile(bind_lite_main(tree, instance), str(path), 'exec'), vars(module))
    original = module._base_toolkit
    def base_factory(*args, **kwargs):
        base = original(*args, **kwargs)
        base.per_round_budget, base.episode_budget = active_call_budgets()
        return base
    module._base_toolkit = base_factory
    module.lite_source_files = ['research/v4/bridge/r3_family_lite.py', 'research/v4/bridge/t1_lite.py']
    return module
