from research.v4.leash_v43.run_manifest import source_custody_enabled
"""One own constructor/provider seam over the original task/lease/outlet code."""
from types import FunctionType, SimpleNamespace

from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle
if source_custody_enabled():
    from research.v4.candidates.r301_v43_same_packet_public_provider_py311_contract_v3_prepare_20260929 import binding as qualified_provider

from .source import HERE, admit, source_text

INSTRUCTION = "Put the item on the left into the microwave and close the door to heat it up."


def prepared_context():
    if not source_custody_enabled():
        return dict(namespace=lite_namespace())
    """Actual classes/globals/closures for Root before comparisons; zero instances."""
    provider_context = qualified_provider.prepared_context()
    provider_context["provider_namespace"].update(admit=admit, source_text=source_text)
    provider_context["sam_namespace"]["source_text"] = source_text
    old_capture = provider_context["guarded_capture"]
    capture = FunctionType(old_capture.__code__, dict(old_capture.__globals__, admit=admit),
                           old_capture.__name__, old_capture.__defaults__, old_capture.__closure__)
    provider_context["provider_namespace"]["_remember_capture"] = capture
    provider_context["guarded_capture"] = capture
    # A delayed source-only compile is consumed by these actual parser/sweep
    # callbacks; it is not a replacement reader, estimator or task executor.
    cache = {}

    def public_functions():
        admit()
        if "functions" not in cache:
            cache["functions"] = provider_context["prepared_functions"]()
        return cache["functions"]

    roles = SimpleNamespace(instance="R3-01", source="moka pot", source_prompt="metal moka pot",
        source_aliases=("moka pot", "moka", "metal pot", "coffee pot", "item"), instruction=INSTRUCTION)

    def observed_sweep(public):
        return public_functions().observed_sweep(roles, public)

    def own_parser(name, parameters, public, locate, held_sweep=None):
        # The original R3 family parser calculates its own public corner bound.
        # Common.resolve applies the same bound to possibly-held UNKNOWN and
        # retains UNKNOWN/geometry ambiguity without changing estimator state.
        return public_functions().proposal(roles, name, parameters, public, locate)

    provider_class = provider_context["provider_class"]
    components = SimpleNamespace(binding=roles,
        bindings=SimpleNamespace(ObservedPayloadSweep=observed_sweep),
        own_provider_class=provider_class)

    def public_provider(actual_components, scene, adapter, sam_client):
        admit()
        if actual_components is not components:
            raise ValueError("exact own R301 components required")
        return provider_class(scene, adapter, sam_client)

    original_constructor = common.Consumer.__init__
    constructor_globals = dict(original_constructor.__globals__, public_provider=public_provider)
    constructor = FunctionType(original_constructor.__code__, constructor_globals,
                               original_constructor.__name__, original_constructor.__defaults__,
                               original_constructor.__closure__)
    constructor.__kwdefaults__ = original_constructor.__kwdefaults__
    original_resolve = common.Consumer.resolve
    resolver_globals = dict(original_resolve.__globals__, proposal_for_p3=own_parser)
    resolver = FunctionType(original_resolve.__code__, resolver_globals, original_resolve.__name__,
                            original_resolve.__defaults__, original_resolve.__closure__)
    resolver.__kwdefaults__ = original_resolve.__kwdefaults__

    class R301OwnBaseConsumer(common.Consumer):
        def __init__(self, *args, components=None, instruction=INSTRUCTION,
                     vla_chunk_controls=None, **kwargs):
            admit()
            if components is not selected_components or instruction != INSTRUCTION:
                raise ValueError("exact own R301 public source/instruction required")
            if vla_chunk_controls is not None:
                raise ValueError("R301 VLA producer completion bound is unverified")
            constructor(self, *args, components=components, instruction=instruction,
                        vla_chunk_controls=None, **kwargs)

        resolve = resolver

    selected_components = components
    consumer = type("R301OwnTaskLeaseConsumer",
                    (TaskConstraints, OwnLeaseLifecycle, R301OwnBaseConsumer), {})
    namespace = SimpleNamespace(**{**vars(common), "Consumer": consumer, "components": components,
        "source_hashes": admit, "provider_context": provider_context,
        "source_qualified": False, "runtime_qualified": False, "native_ready": False,
        "launch_authorized": False, "policy": None, "calibration": None,
        "grasp_assumption": "unverified", "vla_chunk_controls": None})
    return dict(namespace=namespace, Consumer=consumer, components=components,
        provider_context=provider_context, original_constructor=original_constructor,
        constructor=constructor, constructor_globals=constructor_globals,
        original_resolve=original_resolve, resolver=resolver, resolver_globals=resolver_globals,
        public_provider=public_provider, own_parser=own_parser, observed_sweep=observed_sweep,
        public_functions=public_functions)


def namespace():
    if not source_custody_enabled():
        return lite_namespace()
    context = prepared_context()
    admit()
    return context["namespace"]


def lite_namespace():
    """Normal own Consumer construction; no Source supplier or proof chain."""
    from research.v4.candidates.r301_v43_same_packet_public_provider_prepare_20260929 import provider
    from research.v4.candidates.r3_family_method_config_20260927.config import binding
    roles = binding("R3-01")
    functions = provider.prepared_functions()
    def observed_sweep(public):
        return functions.observed_sweep(roles, public)
    def own_parser(name, parameters, public, locate, held_sweep=None):
        return functions.proposal(roles, name, parameters, public, locate)
    components = SimpleNamespace(binding=roles,
        bindings=SimpleNamespace(ObservedPayloadSweep=observed_sweep))
    def public_provider(actual_components, scene, adapter, sam_client):
        return provider.SamePacketR301PublicProvider(scene, adapter, sam_client)
    original_resolve = common.Consumer.resolve
    resolve = FunctionType(original_resolve.__code__,
        dict(original_resolve.__globals__, proposal_for_p3=own_parser), original_resolve.__name__,
        original_resolve.__defaults__, original_resolve.__closure__)
    class OwnBase(common.Consumer):
        def __init__(self, *args, **kwargs):
            kwargs["components"] = components
            kwargs["provider_factory"] = public_provider
            super().__init__(*args, **kwargs)
    OwnBase.resolve = resolve
    consumer = type("R301OwnTaskLeaseConsumer", (TaskConstraints, OwnLeaseLifecycle, OwnBase), {})
    return SimpleNamespace(**{**vars(common), "Consumer": consumer, "components": components,
                              "source_hashes": lambda: {}})
