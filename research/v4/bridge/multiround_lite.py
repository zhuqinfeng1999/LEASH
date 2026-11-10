"""Bind accepted S2/S3 tasks to their original persistent three-round driver."""
import ast
import inspect
import math
import re
import textwrap
from dataclasses import asdict, replace
from types import FunctionType, ModuleType, SimpleNamespace

from research.v4.bridge.swept_lite import public_factory, provider
from research.v4.bridge.t1_lite import public_capture
from research.v4.candidates.s101_v43_round_consumer_20260929 import runtime as rounds
from research.v4.candidates.s101_v43_bridge_round_binding_20260929.bridge import DriverObservationBoundary
from research.v4.candidates.s101_v43_current_core_consumption_v2_20260929.runtime import CoreBindings, OwnerTaskGate
from research.v4.candidates.s1_s2_deferred_production_optin_20260927.hooks import clone_module, session_constructor, boundary_constructor
from research.v4.candidates.s3_v42_tabletop_20260926 import public_gates as original_geometry
from research.v4.candidates.s3_v42_tabletop_20260926 import method_core as original_core
from research.v4.candidates.s3_usage_journal_20260926.session import JournaledS3CodexSession
from research.v4.leash_v43.adapter import V43PublicEnvAdapter
from research.v4.leash_v43.owner_binding import bind_owner_abort
from research.v4.leash_v43.reviewer_transport import ReviewerSession
from research.v4.leash_v43.tool_boundary import ResolvedPrimitive, inject_planner_policy, planner_turn_options
from research.v4.accelerated.p3_gated_toolkit import proposal_for_p3


class RoundConsumer(rounds.S101RoundConsumer):
    """Reuse public payload association; HELD comes only from the estimator."""
    def __init__(self, *args, instructions, sources, **kwargs):
        self.instructions, self.sources = tuple(instructions), tuple(sources)
        self.pending_entity = self.held_entity = self.held_radius = None
        self.last_held = False
        super().__init__(*args, provider_factory=provider, **kwargs)

    def admit_round(self, number, instruction, packet, **kwargs):
        # The same round lifecycle supports the own task's original instructions.
        original = rounds.S101RoundConsumer.admit_round
        admit = FunctionType(original.__code__, {**original.__globals__,
            'INSTRUCTIONS': self.instructions}, original.__name__)
        admit.__kwdefaults__ = original.__kwdefaults__
        self.components.binding.source = self.held_entity or self.sources[number-1]
        return admit(self, number, instruction, packet, **kwargs)

    def resolve(self, name, arguments, workspace):
        public = workspace.public
        kind = self.adapter.estimator.kind.value.lower()
        if kind == 'held' and not self.last_held and self.pending_entity is not None:
            self.held_entity, self.held_radius = self.pending_entity
        if kind == 'empty':
            self.held_entity = self.held_radius = None
            if self.last_held:
                self.pending_entity = None
        if kind != 'unknown':
            self.last_held = kind == 'held'
        envelope = (original_geometry.PayloadEnvelope(self.held_radius)
                    if kind != 'empty' and self.held_radius is not None else None)
        proposal = proposal_for_p3(name, arguments, public,
            self.provider.locate_vla_target, held_sweep=envelope)
        entity = self.held_entity if kind != 'empty' and self.held_entity else proposal.operated_entity
        target = proposal.motion_waypoints[-1] if proposal.motion_waypoints else public.eef_xyz
        near = self.components.near_entity(public, target)
        if kind != 'held' and self.held_entity is None and near and name not in {'release','hold'}:
            entity = near[1]
            box = near[2]
            center = tuple((a+b)/2 for a,b in zip(box.lower,box.upper))
            self.pending_entity = (entity, math.dist(target,center)+math.dist(box.lower,box.upper)/2)
        if kind != 'held' and name in {'pi0_pick','pi0_doubled'}:
            self.pending_entity = None
            prompt = self.components.canonical_prompt(str(arguments.get('prompt','')))
            named = [n for n in self.components.movable if re.search(r'\b'+re.escape(n)+r'\b',prompt)]
            if named:
                source = min(named, key=prompt.find)
                box = public.box(source)
                if box:
                    self.pending_entity = (source,math.dist(box.lower,box.upper)/2+.06)
                entity = source
        if entity in self.components.movable:
            self.components.binding.source = entity
        proposal = replace(proposal, operated_entity=entity,
            geometry_ambiguous=proposal.geometry_ambiguous or kind == 'unknown'
                or (kind == 'held' and envelope is None))
        self._public_phase = None
        self._version(workspace)
        self.proposals.append(dict(name=name,arguments=dict(arguments),proposal=asdict(proposal),
            holding_source='executed_records_and_public_proprio'))
        return ResolvedPrimitive(workspace.version,proposal,
            entities=tuple(x for x in (entity,) if x),geometry_unknown=proposal.geometry_ambiguous)


def own_components(instance):
    from research.v4.candidates.s1_multiround_driver_20260925 import bridge, native_wiring
    if instance.startswith('S2-'):
        from research.v4.candidates.s2_future_production_dispatch_20260927.factory import binding
        from research.v4.candidates.s2_future_method_v2_20260927.vocabulary import vocabulary
        from research.v4.candidates.s2_higher_method_20260926.method_core import strict_label
        spec = binding(instance)
        prompts, aliases = vocabulary(instance)
        instructions, sources = spec.instructions, ('pill','ketchup','pill')
        class Scene(spec.facade_class):
            capture_agent = public_capture
        from research.v4.bridge.s2_lite_scoring import CurrentS2Scorer
        scorer = CurrentS2Scorer
        movable = frozenset({'pill','ketchup'})
    elif instance.startswith('S3-'):
        from research.v4.candidates.user_rework_support_20260927.s3_family.scene import PlanarFamilyFacade, CONFIGS
        from research.v4.candidates.s3_physical_support_predicate_20260926.scorer import PhysicalSupportS3Scorer
        prompts, aliases = original_geometry.PROMPTS, original_geometry.ALIASES
        instructions, sources = CONFIGS[instance][1], ('plate','ketchup','cookie')
        class Scene(PlanarFamilyFacade):
            capture_agent = public_capture
            def __init__(self, *args, **kwargs):
                super().__init__(*args, instance=instance, **kwargs)
        scorer, movable = PhysicalSupportS3Scorer, original_geometry.MOVABLE
        strict_label = original_core.strict_label
    else:
        raise ValueError('Only accepted S2/S3 own tasks')
    def canonical_prompt(prompt):
        cleaned = re.sub(r'[^a-z0-9]+',' ',prompt.casefold()).strip()
        pattern = r'\b(?:'+'|'.join(re.escape(x) for x in sorted(aliases,key=lambda x:(-len(x),x)))+r')\b'
        return re.sub(pattern,lambda m:aliases[m.group(0)],cleaned)
    def locate(prompt, public):
        text = canonical_prompt(prompt)
        named = [n for n in prompts if re.search(r'\b'+re.escape(n)+r'\b',text)]
        if not named:
            return None
        entity = max(named,key=text.rfind)
        box = public.box(entity)
        return None if box is None else (entity,tuple((a+b)/2 for a,b in zip(box.lower,box.upper)))
    near = FunctionType(original_geometry.S3ActionBoundary._near_entity.__code__,
        {**original_geometry.__dict__,'MOVABLE':movable}, '_near_entity')
    return SimpleNamespace(Scene=Scene,Scorer=scorer,instructions=tuple(instructions),
        sources=sources,binding=SimpleNamespace(source=sources[0]),movable=movable,
        PublicWorkspaceFactory=public_factory(prompts,locate),
        near_entity=near,canonical_prompt=canonical_prompt,bridge=bridge,wiring=native_wiring,
        strict_label=strict_label)


def session_type(method, policy):
    """Original model session, same tools and model; add only Ours policy."""
    original = JournaledS3CodexSession.__aenter__
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    class Policy(ast.NodeTransformer):
        def visit_Assign(self,node):
            node = self.generic_visit(node)
            if ast.unparse(node.targets[0]) == 'self._planner' and method != 'B0':
                return [node, ast.parse("inject_planner_policy(self._planner,method="+repr(method)+",approved_policy=policy,expected_policy_sha256=policy.sha256)").body[0]]
            if ast.unparse(node.targets[0]) == 'self.thread_starts':
                return [node,ast.parse('self._planner._turn_options = planner_turn_options(self._planner)').body[0]]
            return node
    ns = {**original.__globals__,'inject_planner_policy':inject_planner_policy,
          'planner_turn_options':planner_turn_options,'policy':policy}
    exec(compile(ast.fix_missing_locations(Policy().visit(tree)),__file__,'exec'),ns)
    return type('OwnPolicySession',(JournaledS3CodexSession,),{'__aenter__':ns['__aenter__']})


def core_module(components, policy, *, control_method=None, calibration=None):
    from research.v4.bridge.run_b0 import active_call_budgets, lite_public_toolkit
    holder = {'consumer':None}
    context = CoreBindings(SimpleNamespace(own_components=components),
        policy=policy,calibration=None,source_hashes=lambda:{})
    if control_method is not None:
        from research.v4.bridge.multiround_control_lite import control_context
        context = control_context(components, method=control_method,
                                  policy=policy, calibration=calibration)
    def make_consumer(base,adapter,scene,sam,reviewer,episode):
        if control_method is None:
            consumer = RoundConsumer(base,adapter,components=components,scene=scene,sam_client=sam,
                reviewer=reviewer,policy=policy,instruction=scene.instruction,episode=episode,
                primitives=base.original._primitives,vla_chunk_controls=5,
                instructions=components.instructions,sources=components.sources)
        else:
            context.vla_chunk_controls = 5 if __import__('os').environ['SAFEVLA_LEASH_BACKEND']=='live' else None
            consumer = context.make_consumer(base, adapter, scene, sam, reviewer,
                                             scene.instruction, episode)
        holder['consumer'] = consumer
        return consumer
    def wrapper(inner,*,task_gate):
        original = original_core.S3PersistentWrapper.run_round
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        class Seal(ast.NodeTransformer):
            def visit_Expr(self,node):
                node = self.generic_visit(node)
                if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='self.history.append':
                    return [node,ast.parse("seal_consumer(self.history[-1],seal,outcome)").body[0]]
                return node
        def seal_consumer(row,boundary,outcome):
            consumer=holder['consumer']
            if consumer is not None:
                if control_method is None:
                    consumer.seal_round(primitive_summaries=row.primitive_summaries,
                        boundary=boundary,resource_end=outcome.resource_end)
                else:
                    context.seal_round(consumer, primitive_summaries=row.primitive_summaries,
                        boundary=boundary, resource_end=outcome.resource_end)
        ns={**original.__globals__,'INSTRUCTIONS':components.instructions,'seal_consumer':seal_consumer}
        exec(compile(ast.fix_missing_locations(Seal().visit(tree)),__file__,'exec'),ns)
        selected=type('OwnRoundWrapper',(original_core.S3PersistentWrapper,),{'run_round':ns['run_round']})
        return selected(inner,task_gate=task_gate)
    module = clone_module(original_core, INSTRUCTIONS=components.instructions,
        strict_label=components.strict_label,
        PublicLatchToolkit=lite_public_toolkit,ReviewerSession=ReviewerSession,
        S3CodexSession=session_constructor(session_type('Ours',policy)),
        S3PersistentWrapper=boundary_constructor(wrapper))
    module.strict_label = components.strict_label
    source = inspect.getsource(original_core.run_s3_core)
    class Bind(ast.NodeTransformer):
        def visit_If(self,node):
            condition=ast.unparse(node.test)
            if control_method is not None and condition == "method not in {'B0', 'Ours'}":
                node.test = ast.parse('method != '+repr(control_method),mode='eval').body
            if control_method is not None and condition == "method == 'Ours'":
                node.test = ast.parse('method in '+repr(('B2','B3')),mode='eval').body
            if condition.startswith('not diagnostic and') and ('R2V42Adapter' in condition or 'P3GatedAdapter' in condition):
                return None
            if 'P3GatedAdapter' in condition:
                node.test=ast.parse("reviewer is None or not isinstance(adapter,V43PublicEnvAdapter)" if control_method else "reviewer is None or sam3_client is None or not isinstance(adapter,V43PublicEnvAdapter)",mode='eval').body
            if condition=="(b0_toolkit.per_round_budget, b0_toolkit.episode_budget) != (48, 144)":
                node.test=ast.parse('(b0_toolkit.per_round_budget,b0_toolkit.episode_budget) != active_call_budgets()',mode='eval').body
            if condition=='action':
                return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            node=self.generic_visit(node)
            target=ast.unparse(node.targets[0])
            if target in {'workspace','task_gate'}:
                node.value=ast.Constant(None)
            elif target=='action':
                active = repr(control_method or 'Ours')
                return ast.parse("consumer = make_consumer(public,adapter,scene,sam3_client,reviewer,episode_id) if method=="+active+" else None\naction = consumer.boundary if consumer else None").body
            elif target=='fault':
                node.value=ast.parse('FaultStopToolkit(context.observe_boundary(consumer) if consumer else public,adapter)',mode='eval').body
                return [node,ast.parse("pending = {'value':None}").body[0]]
            elif target=='callbacks':
                node.value=ast.parse("S1OwnerCallbacks(fault,capture=(lambda:context.admit_on_owner(consumer,scene,pending)) if consumer else lambda:None,recover=lambda:None,advance=advance)",mode='eval').body
            elif target=='owner':
                return [node,ast.parse("if consumer and method!='B1':bind_owner_abort(owner,adapter)").body[0]]
            elif target=='task_adapter':
                node.value=ast.parse("OwnerTaskGate(context,consumer,pending,owner,callbacks) if consumer else S1TaskGateAdapter(method=method,task_gate=None,owner_proxy=owner,owner_callbacks=callbacks,episode_id=episode_id)",mode='eval').body
            elif target=='action_fault':
                node.value=ast.parse("bool(consumer and any(x.get('system_failure') for x in consumer.action_report()['logs']))",mode='eval').body
            elif target=='commit_fault':
                node.value=ast.parse("any(x.get('system_failure') for x in adapter.dispatcher.logs) if consumer and method!='B1' else False",mode='eval').body
            return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node.func)=='S3CodexSession':
                node.func=ast.parse('session_constructor(session_type(method,policy))',mode='eval').body
            return node
        def visit_Dict(self,node):
            node=self.generic_visit(node)
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='action_logs':
                    node.values[i]=ast.parse("consumer.action_report()['logs'] if consumer else []",mode='eval').body
                elif isinstance(key,ast.Constant) and key.value=='action_proposals':
                    node.values[i]=ast.parse('consumer.proposals if consumer and hasattr(consumer,"proposals") else []',mode='eval').body
                elif isinstance(key,ast.Constant) and key.value=='commit_decisions':
                    node.values[i]=ast.parse('adapter.dispatcher.logs if consumer and method!="B1" else []',mode='eval').body
            return node
    module.__dict__.update(make_consumer=make_consumer,CoreBindings=CoreBindings,
        DriverObservationBoundary=DriverObservationBoundary,OwnerTaskGate=OwnerTaskGate,
        context=context,V43PublicEnvAdapter=V43PublicEnvAdapter,bind_owner_abort=bind_owner_abort,
        active_call_budgets=active_call_budgets,session_type=session_type,
        session_constructor=session_constructor,policy=policy)
    exec(compile(ast.fix_missing_locations(Bind().visit(ast.parse(source))),__file__,'exec'),vars(module))
    module.holder=holder
    return module


def runner(instance, *, control_plan=None):
    from research.v4.bridge.run_b0 import ROOT,lite_runtime,bind_lite_main
    components=own_components(instance)
    bindings=lite_runtime(instance,components)
    method = None if control_plan is None else control_plan.method
    if control_plan is not None:
        from research.v4.familiarization.controls_preparation.control_entry import runtime_overrides
        bindings.update(runtime_overrides(control_plan, bindings))
    core=core_module(components,bindings['V43_POLICY'], control_method=method,
                     calibration=bindings['V43_CALIBRATION'])
    family=instance.split('-')[0].lower()
    package=('research.v4.candidates.s2_higher_method_20260926' if family=='s2'
             else 'research.v4.candidates.s3_v42_tabletop_20260926')
    path=ROOT/(package.replace('.','/')+'/run_method.py')
    class Bind(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module in {'.rpent_facade','rpent_facade'}:
                return ast.parse(node.names[0].name+' = OWN_SCENE').body
            if node.module in {'.offline_scorer','offline_scorer'}:
                return ast.parse(node.names[0].name+' = OWN_SCORER').body
            if node.module in {'.method_core','method_core'}:
                return ast.parse(node.names[0].name+' = OWN_CORE').body
            return node
        def visit_If(self,node):
            if ast.unparse(node.test)=="a.method == 'Ours'" and any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reviewer' for t in n.targets) for n in node.body):
                node.body=ast.parse("reviewer,adapter = V43.make_reviewer_adapter(scene,instruction=scene.instruction,model=a.reviewer_model_id or a.model_id,episode=output.name,output=output,policy=V43_POLICY,calibration=V43_CALIBRATION,on_step=None)").body
                if method is not None:
                    node.test=ast.parse('a.method == '+repr(method if method!='B1' else 'Ours'),mode='eval').body
            return self.generic_visit(node)
    module=ModuleType(__name__+'._'+instance.replace('-','_'));module.__package__=package
    module.__file__=str(path)
    module.__dict__.update(bindings,OWN_SCENE=components.Scene,OWN_SCORER=components.Scorer,
        OWN_CORE=core.run_s3_core)
    exec(compile(bind_lite_main(Bind().visit(ast.parse(path.read_text())),instance),str(path),'exec'),vars(module))
    module.INSTRUCTIONS=components.instructions
    module.PROTOCOL='LEASH-v4.3-custody-off-own-'+instance
    module.hashes=lambda:{}
    module.current_qualification=lambda:True
    module.lite_source_files=['research/v4/bridge/multiround_lite.py',
        'research/v4/bridge/swept_lite.py','research/v4/bridge/t1_lite.py',
        'research/v4/candidates/s101_v43_round_consumer_20260929/runtime.py',
        'research/v4/candidates/s3_v42_tabletop_20260926/method_core.py']
    if control_plan is not None:
        module.lite_source_files += ['research/v4/bridge/multiround_control_lite.py',
                                    'research/v4/bridge/s1_control_lite.py']
    if instance.startswith('S2-'):
        module.lite_source_files += ['research/v4/bridge/s2_lite_scoring.py',
            'research/v4/candidates/s2_01_upper_neck_122_20260927/evaluator.py']
    module.own_components, module.lite_core = components,core
    return module
