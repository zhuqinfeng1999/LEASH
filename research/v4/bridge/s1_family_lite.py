"""Own S1 nouns over the existing two-round Owner, recovery and public reader."""
import ast
import inspect
import textwrap
from types import FunctionType, SimpleNamespace

from research.v4.candidates.s1_own_method_dispatch_20260927 import factory as semantic
from research.v4.candidates.s101_v43_public_provider_20260929 import public
from research.v4.candidates.s101_v43_own_public_consumer_20260929 import runtime as original
from research.v4.candidates.s101_v43_round_consumer_20260929 import runtime as rounds
from research.v4.candidates.s101_v43_recovery_literal_regions_20260929.runtime import RecoveryScope
from research.v4.candidates.s101_v43_actual_core_binding_20260929.recovery import RecoveryScope as PhysicalRecovery
from research.v4.candidates.s101_v43_current_core_consumption_v2_20260929.runtime import CoreBindings
from research.v4.leash_v43.tool_boundary import GeometryDecision


def noun_function(function, binding):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = semantic.labels(node.value, binding)
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
    return namespace[function.__name__]


def public_components(instance):
    current = semantic.components(instance)
    # The existing same-packet typed reader already binds its parser's text to
    # the supplied own Binding. Select that existing callback for this instance.
    select = FunctionType(public.selected_components.__code__,
        {**public.selected_components.__globals__,
         'own':SimpleNamespace(components=lambda _instance:current)},
        public.selected_components.__name__)
    selected = select()
    selected.binding = SimpleNamespace(source=current.binding.noun)
    return selected


def consumer_type(instance):
    selected = public_components(instance)
    source = selected.binding.source
    instructions = selected.original.instructions
    sweep_init = original.ExistingSweep.__init__
    sweep = type('OwnS1PayloadSweep', (original.ExistingSweep,), {'__init__':FunctionType(
        sweep_init.__code__, {**sweep_init.__globals__, 'SOURCE':source},
        sweep_init.__name__, sweep_init.__defaults__, sweep_init.__closure__)})
    resolver = noun_function(original.S101OwnPublicConsumer.resolve, selected.original.binding)
    recover = noun_function(PhysicalRecovery.recover_on_owner, selected.original.binding)
    admit = rounds.S101RoundConsumer.admit_round
    admitted = FunctionType(admit.__code__, {**admit.__globals__, 'INSTRUCTIONS':instructions},
        admit.__name__, admit.__defaults__, admit.__closure__)
    admitted.__kwdefaults__ = admit.__kwdefaults__

    class OwnConsumer(RecoveryScope, original.S101OwnPublicConsumer):
        def __init__(self, *args, **kwargs):
            self.own_public, self.ObservedPayloadSweep, self.role_logs = selected, sweep, []
            def provider(_components, scene, adapter, sam_client):
                return public.make_provider(scene=scene, adapter=adapter, sam3_client=sam_client,
                    forbidden_entities=self.current_public_entities, components=selected)
            kwargs.update(components=selected, provider_factory=provider)
            # RecoveryScope's cooperative initializer still establishes its
            # lifecycle; only the old pan-only constructor is bypassed.
            self._recovery_context, self._recovery_number, self.recovery_logs = None, 0, []
            rounds.S101RoundConsumer.__init__(self, *args, **kwargs)
        admit_round = admitted
        resolve = resolver
        def geometry(self, resolved, workspace, lease):
            proposal = resolved.proposal
            if (self.adapter.estimator.kind.value.lower() != 'empty'
                    and workspace.public.box(source) is None
                    and (proposal.motion_waypoints or proposal.name == 'vla_act')):
                return GeometryDecision(False, 'Current public '+source+' payload envelope is unavailable')
            return rounds.S101RoundConsumer.geometry(self, resolved, workspace, lease)
        def recover_on_owner(self, **kwargs):
            if (self._validated_task or {}).get('result') in {'ASK','REFUSE'}:
                return RecoveryScope.recover_on_owner(self, **kwargs)
            return recover(self, **kwargs)
    return OwnConsumer


def core_context(parent, *, policy, calibration, source_hashes):
    cls = consumer_type(parent.own_components.binding.instance)
    class Context(CoreBindings):
        def make_consumer(self, public_base, adapter, scene, sam_client, reviewer, instruction, episode):
            consumer = cls(public_base, adapter, scene=scene, sam_client=sam_client,
                reviewer=reviewer, policy=self.policy, instruction=instruction,
                episode=episode, primitives=public_base.original._primitives,
                vla_chunk_controls=self.vla_chunk_controls)
            self.last_consumer = consumer
            self.operational_calls.append('make_consumer')
            return consumer
    return Context(parent, policy=policy, calibration=calibration, source_hashes=source_hashes)
