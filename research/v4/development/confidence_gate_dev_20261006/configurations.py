"""Direct accepted physics/scorer selection for the new confidence entry.

These bindings construct no scene and perform no model request. They import
physical constructors and private evaluators, never a retired Consumer or a
geometry gate. Only the runner keeps the private evaluator; gate evidence is
still built from the two public cameras and public robot state.
"""
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import json
import os
import uuid

from research.v4.bridge.t1_lite import public_capture

ROOT = Path(__file__).resolve().parents[4]
PUBLIC_VIEWS = ('agentview', 'robot0_eye_in_hand')
BASE_REPRESENTATIVES = ('P2-01', 'P3-01', 'P4-01', 'E1-01', 'E2-01', 'E3-01',
                        'I1-01', 'I2-01', 'I3-01', 'I4-01', 'S1-01', 'S2-01', 'S3-01')


def _qualified_initial_scene(parent, instance):
    """Use the existing empty-reset source only for accepted fresh recipes.

    This constructor reads configuration, never dynamic object/assist truth.
    Unlisted instances/seeds retain their ordinary UNKNOWN start.
    """
    from research.v4.leash_v43.initial_occupancy import VERSION, empty_reset_fact
    qualification=json.loads((ROOT/'research/v4/results/confidence_gate_dev_20261006/initialization_qualification.json').read_text())
    recipe=qualification.get('additional_recipes',{}).get(instance)
    if recipe is None:
        return parent, ()
    if qualification.get('protocol')!=VERSION or recipe.get('fresh_constructor_only') is not True:
        raise ValueError('Invalid accepted empty-reset configuration')
    seeds=tuple(recipe['accepted_seeds'])
    class FreshScene(parent):
        def __init__(self, seed=0, **kwargs):
            self._public_initial_fact=None
            super().__init__(seed, **kwargs)
            if seed not in seeds:
                return
            if 'constructor_factory' in recipe:
                same_recipe=(getattr(self, 'initial_reset_recipe', None)
                             ==recipe['constructor_factory'])
            else:
                same_recipe=(self.layout_version==recipe['layout']
                             and self.initial_reset_revision==recipe['reset_revision'])
            if self.sequence!=0 or not same_recipe:
                self.close()
                raise ValueError('Fresh constructor does not match its accepted empty-reset recipe')
            self._public_initial_fact=empty_reset_fact(
                episode=Path(os.environ['SAFEVLA_RUN_OUTPUT']).name,
                reset_id=uuid.uuid4().hex,initial_step=0,fresh=True,recipe_qualified=True)

        def public_initial_occupancy(self):
            return None if self._public_initial_fact is None else dict(self._public_initial_fact)

        def invalidate_initial_occupancy(self):
            self._public_initial_fact=None

        def capture_agent(self):
            packet=public_capture(self)
            fact=self.public_initial_occupancy()
            if fact is not None:
                packet['robot_state']['initial_occupancy']=fact
            return packet
    return FreshScene, seeds


@dataclass(frozen=True)
class PublicBinding:
    instance: str
    instruction: str
    source: str | None = None
    target: str | None = None


def registry_row(instance):
    rows = json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']
    row = next((row for row in rows if row['id'] == instance), None)
    if row is None:
        raise ValueError('Unregistered instance: '+instance)
    return row


def resolve(path):
    module, name = path.split(':')
    return getattr(import_module(module), name)


def _configured(instance, parent, evaluator, *, source=None, target=None,
                instructions=None, evaluator_requires_scene=False,
                original_binding=None, pending=()):
    row = registry_row(instance)
    instructions = tuple(row['instructions']) if instructions is None else tuple(instructions)
    if instructions != tuple(row['instructions']):
        raise ValueError('Accepted instruction/registry mismatch for '+instance)
    if tuple(getattr(parent, 'public_views', PUBLIC_VIEWS)) != PUBLIC_VIEWS:
        raise ValueError('Physical binding changed public camera protocol')
    persistent = len(instructions) > 1
    class Scene(parent):
        # Unlike some old facades, this exact public builder neither increments
        # frame IDs on reads nor injects the private grasp latch into packets.
        capture_agent = public_capture
        public_views = PUBLIC_VIEWS
    if not persistent:
        # Own-family factories previously applied this same registered text
        # substitution after construction. No physical parameter changes.
        Scene.instruction = instructions[0]
        def init(self, *args, **kwargs):
            parent.__init__(self, *args, **kwargs)
            self.instruction = instructions[0]
        Scene.__init__ = init
    Scene.__name__ = instance.replace('-', '')+'ConfidencePublicScene'
    if instance in ('P2-01', 'P2-02', 'P2-03', 'P2-04', 'P2-05', 'P2-06',
                    'P3-01', 'E3-01', 'S1-01'):
        # Metadata for the independently checked, unchanged fresh constructor.
        # Qualification below restricts issuance to the accepted seed/recipe;
        # neither this identity nor the public initializer reads object truth.
        Scene.initial_reset_recipe = f'{parent.__module__}:{parent.__qualname__}:{instance}'
    Scene, qualified_seeds = _qualified_initial_scene(Scene, instance)
    binding = PublicBinding(instance, instructions[0], source, target)
    return SimpleNamespace(binding=binding, original_binding=original_binding,
        Scene=Scene, AcceptedScene=parent, Evaluator=evaluator, instructions=instructions,
        persistent=persistent, evaluator_requires_scene=evaluator_requires_scene,
        initial_occupancy_source='verified_empty_reset' if qualified_seeds else None,
        empty_reset_qualified=bool(qualified_seeds), qualified_empty_seeds=qualified_seeds,
        physical_binding_ready=True, native_confidence_verified=False,
        pending=tuple(pending))


def configuration(instance):
    """Select existing own physical bindings; this function starts no run."""
    if instance in ('R1-01','R5-01'):
        from .r15_configuration import configuration as r15_configuration
        return r15_configuration(instance)
    if instance in ('R2-01','R3-01','R4-01'):
        from .r234_configuration import configuration as r234_configuration
        return r234_configuration(instance)
    if instance in ('P1-04','P1-05','P1-06'):
        from research.v4.candidates.p1_own_plain_config_20260927.config import components
        b,parent,evaluator=components(instance)
        if instance=='P1-05':
            # Existing method carrier keeps MULTICCD16 physics and removes only
            # the old offline qualification's control/wall/logging caps.
            from research.v4.candidates.p1_05_method_carrier_20260927.scene import Scene as parent
        elif instance=='P1-06':
            # API aliases only; the own full-shape/support predicates stay the same.
            from .p106_private_data_compat import P1BucketEvaluator as evaluator
        return _configured(instance,parent,evaluator,source=b.source,target=b.target,
                           original_binding=b)
    if instance.startswith('P2-'):
        from research.v4.candidates.p2_own_plain_config_20260927.config import components
        b, parent, evaluator = components(instance)
        if instance == 'P2-01':
            # Current revised P201 factory selects these same own constructors;
            # its old Consumer/factory parent is intentionally not imported.
            from .p2_contact_zone import Scene as parent
            from research.v4.candidates.user_rework_p2_20260927.evaluator_v2 import Evaluator as evaluator
            b = replace(b, scene='research.v4.development.confidence_gate_dev_20261006.p2_contact_zone:Scene',
                evaluator='research.v4.candidates.user_rework_p2_20260927.evaluator_v2:Evaluator')
        from .p2_front_placement_v4 import adapt_configuration
        return adapt_configuration(_configured(instance, parent, evaluator,
            source=b.source,target=b.target,original_binding=b))
    if instance.startswith('P3-'):
        from research.v4.candidates.p3_own_method_dispatch_20260927.config import binding
        from research.v4.accelerated.p3_offline_scoring import P3V42Evaluator
        b, instruction, parent = binding(instance)
        if instance == 'P3-01':
            from .p3_sampled_rest import P3SampledRestEvaluator as P3V42Evaluator
        return _configured(instance,parent,P3V42Evaluator,source=b.source,target=b.target,
                           instructions=(instruction,),original_binding=b)
    if instance.startswith('P4-'):
        from research.v4.candidates.user_revision_real_p4_v4_20260928.config import own_components
        b,parent,evaluator=own_components(instance)
        if instance == 'P4-01':
            from .p4_physical_completion import (
                P401PhysicalCompletionScene as parent,
                P401PhysicalCompletionEvaluator as evaluator,
            )
        return _configured(instance,parent,evaluator,source=b.source,target=b.target,original_binding=b)
    if instance.startswith('E1-'):
        from research.v4.candidates.e1_future_public_binding_20260927.config import resolve_components
        b,parent,evaluator=resolve_components(instance)
        recipe=f'{parent.__module__}:{parent.__qualname__}:{instance}'
        class E1Parent(parent):
            def __init__(self, seed=0, **kwargs):
                super().__init__(seed, **kwargs)
                self.initial_reset_recipe=recipe
        return _configured(instance,E1Parent,evaluator,source=b.source,target=b.target,original_binding=b)
    if instance.startswith('E2-'):
        from research.v4.candidates.e2_native_event_20260926.evaluator import E2Evaluator
        if instance=='E2-01':
            from research.v4.candidates.e2_cat_corrected_hold_20260926.facade import E2CorrectedHoldScene as parent
            source,target='dictionary','cardboard box'
            b=None
        elif instance=='E2-02':
            from research.v4.candidates.e2_laundry_cat_20260926.facade import E2LaundryScene as parent
            source,target='dumbbell','laundry basket'
            b=None
        else:
            from research.v4.candidates.e2_dog_own_method_dispatch_20260927.config import binding
            b=binding(instance)
            parent=getattr(import_module(b.scene_module),b.scene_class)
            source,target=b.source,b.target
        recipe=f'{parent.__module__}:{parent.__qualname__}:{instance}'
        if instance == 'E2-01':
            from .e2_physical_event import E2PhysicalEventMixin, E2PhysicalEventEvaluator
            class PhysicalParent(E2PhysicalEventMixin, parent):
                pass
            parent = PhysicalParent
            E2Evaluator = E2PhysicalEventEvaluator
        class E2Parent(parent):
            def __init__(self, seed=0, **kwargs):
                super().__init__(seed, **kwargs)
                self.initial_reset_recipe=recipe
        configured = _configured(instance,E2Parent,E2Evaluator,source=source,target=target,original_binding=b)
        if instance == 'E2-01':
            from .e2_physical_event import REVISION
            configured.scene_event_revision = REVISION
        return configured
    if instance.startswith('E3-'):
        from research.v4.candidates.e3_family_current_method_config_20260927.config import get
        spec=get(instance)
        parent,original,snapshot=resolve(spec.factory),resolve(spec.scorer),resolve(spec.snapshot)
        original_recipe=f'{parent.__module__}:{parent.__qualname__}:{instance}'
        class Evaluator(original):
            def observe(self,scene):
                # An owner-side private scorer adapter, never gate evidence.
                return super().observe(snapshot(scene))
        if instance == 'E3-01':
            from .e3_physical_event import E3PhysicalEventMixin, E3PhysicalEventEvaluator, REVISION
            class PhysicalParent(E3PhysicalEventMixin, parent):
                pass
            parent, Evaluator = PhysicalParent, E3PhysicalEventEvaluator
        configured = _configured(instance,parent,Evaluator,instructions=(spec.instruction,),original_binding=spec)
        if instance == 'E3-01':
            # The mixin changes only the executed-control event/scorer boundary.
            # Its unchanged fresh constructor retains the accepted recipe;
            # qualification still checks the real initial sequence and seed.
            configured.Scene.initial_reset_recipe = original_recipe
            configured.scene_event_revision = REVISION
        return configured
    if instance.startswith(('I1-','I2-','I3-','I4-')):
        if instance not in {f'I{family}-{number:02}' for family in range(1,5) for number in range(1,4)}:
            raise ValueError('Unregistered accepted I binding')
        if instance.startswith('I4-'):
            from research.v4.candidates.i4_common_park_readable_tag_20260927.scene import create
        elif instance.endswith('-02'):
            from research.v4.candidates.user_rework_support_20260927.i_candle_front_v2.scene import create
        else:
            from research.v4.candidates.user_rework_support_20260927.i_family.scene import create
        from research.v4.candidates.i1_v42_t1t7_20260925.facade import I1Facade
        from research.v4.candidates.i_common_floor_completion_20260926.evaluator import KnifeEvaluator,NamedEvaluator
        import numpy as np
        class Parent(I1Facade):
            def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=0):
                self.backend,self.grasp_assist=create(instance,seed,
                    camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
                self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend)
                self.raw=self.backend._get_observations(force_update=True)
                self.sequence,self.assist_events=0,[]
                # Static identity of the fresh factory actually invoked above.
                # This is not an object-state/assist query or an EMPTY finding.
                self.initial_reset_recipe=f'{create.__module__}:{create.__qualname__}:{instance}'
        old=KnifeEvaluator if instance.endswith('-01') else NamedEvaluator
        class Evaluator(old):
            def __init__(self):super().__init__(instance)
        return _configured(instance,Parent,Evaluator,source='ketchup')
    if instance.startswith('S1-'):
        from research.v4.candidates.s1_own_method_dispatch_20260927.config import binding
        from research.v4.candidates.s1_v42_tabletop_20260925.evaluator import S1Evaluator
        b,instructions,parent=binding(instance)
        class Evaluator(S1Evaluator):
            def finish(self,*args,**kwargs):
                return {**super().finish(*args,**kwargs),'task_id':instance}
        return _configured(instance,parent,Evaluator,source=b.noun,instructions=instructions,
            original_binding=b,pending=('Retain one persistent planner session; start_round evaluator sequencing and finish/seal driver seam not yet installed in confidence entry.',))
    if instance.startswith('S2-'):
        paths={
            'S2-01':'research.v4.candidates.s2_01_seeded_yaw_20260927.scene:Scene',
            'S2-02':'research.v4.candidates.s2_02_release_yaw_20260927.scene:Scene',
            'S2-03':'research.v4.candidates.s2_03_release_yaw_20260927.scene:Scene',
            'S2-04':'research.v4.candidates.s2_04_release_yaw_20260927.scene:Scene',
            'S2-05':'research.v4.candidates.s2_05_release_yaw_20260927.scene:Scene'}
        if instance not in paths:raise ValueError('Unregistered accepted S2 binding')
        physical=resolve(paths[instance])
        from research.v4.candidates.s2_higher_method_20260926.rpent_facade import S2RPentFacade
        from research.v4.bridge.s2_lite_scoring import CurrentS2Scorer
        instructions=tuple(registry_row(instance)['instructions'])
        class Parent(S2RPentFacade):
            public_views=PUBLIC_VIEWS
            def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=0):
                super().__init__(seed,camera_observations=camera_observations,
                    render_gpu_device_id=render_gpu_device_id,diagnostic_scene_factory=physical)
                if instance == 'S2-01':
                    # The independently accepted recipe includes the original
                    # forty zero-command constructor settling steps. This is
                    # the first business step, not a reset of physics time.
                    self.initial_reset_recipe = paths[instance]+':S2-01:fixed40'
            @property
            def instruction(self):return instructions[self.round_number-1]
            def advance_round(self):
                self._scene.start_round(self.round_number+1)
                return self.instruction
        return _configured(instance,Parent,CurrentS2Scorer,instructions=instructions,
            evaluator_requires_scene=True,
            pending=('Persistent planner/three-round finish/seal bridge required; CurrentS2Scorer(scene) must attach the original private evaluator before any control.',))
    if instance.startswith('S3-'):
        from research.v4.candidates.user_rework_support_20260927.s3_family.scene import PlanarFamilyFacade,CONFIGS
        from research.v4.candidates.s3_physical_support_predicate_20260926.scorer import PhysicalSupportS3Scorer
        if instance not in CONFIGS:raise ValueError('Unregistered accepted S3 binding')
        class Parent(PlanarFamilyFacade):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,instance=instance,**kwargs)
                self.initial_reset_recipe=f'{PlanarFamilyFacade.__module__}:{PlanarFamilyFacade.__qualname__}:{instance}'
        return _configured(instance,Parent,PhysicalSupportS3Scorer,instructions=CONFIGS[instance][1],
            pending=('Retain one persistent planner/three-round finish/seal bridge and original scene.advance_round; not yet installed in confidence entry.',))
    raise ValueError('No confidence physical binding for '+instance)


def readiness(instance):
    """Ordinary configuration metadata; no runtime qualification is inferred."""
    c=configuration(instance)
    return dict(instance=instance,instructions=list(c.instructions),
        accepted_scene=f'{c.AcceptedScene.__module__}:{c.AcceptedScene.__qualname__}',
        evaluator=f'{c.Evaluator.__module__}:{c.Evaluator.__qualname__}',
        physical_binding_ready=c.physical_binding_ready,
        native_confidence_verified=False,persistent=c.persistent,
        evaluator_requires_scene=c.evaluator_requires_scene,
        empty_reset_qualified=False,pending=list(c.pending))
