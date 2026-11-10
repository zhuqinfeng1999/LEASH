"""Opt-in finite-transition entry for explicitly registered development/formal runs."""
import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from .scope import TransitionDispatcher, VERSION, digest, prepared_motion_scope
from research.v4.development.g3_event_count_repair_20261020.registry import (
    CountBoundTransitionRegistry as TransitionRegistry, VERSION as EVENT_COUNT_REPAIR_VERSION,
)
from research.v4.development.g3_event_count_repair_20261020.estimator import CompletedOpenHoldingEstimator
from .reviewer import TransitionLunaWorker, TransitionScoreSession

ROOT=Path(__file__).resolve().parents[4]
PACKAGE=Path(__file__).resolve().parent
ENTRYPOINT='research.v4.development.g3_transition_dev_20261019.native'


def validate_registration(args,plan,row):
    matrix=plan.get('optimized_matrix')
    if (args.method!='Ours-conf' or (not matrix and args.reviewer_backend!='luna')
            or row.get('transition_version')!=VERSION):
        raise ValueError('Only an explicitly registered finite-transition condition is allowed')
    if matrix:
        from research.v4.bridge.formal_optimized_matrix import validate_scope, arm_key, arm_enabled
        validate_scope(plan)
        if (row.get('phase')!='formal' or not args.formal
                or arm_key(row) not in matrix['arms'] or not arm_enabled(plan,row)
                or row.get('reviewer_backend')!=args.reviewer_backend):
            raise ValueError('Explicit enabled matrix model/backend registration required')
    if row.get('phase')=='formal':
        if (not args.formal or not plan.get('g3_transition_rerun')
                or plan.get('native_entrypoint')!=ENTRYPOINT
                or plan.get('authorization',{}).get('state')!='RUNNING'
                or (not matrix and row.get('planner_model')!='gpt-6-astra')):
            raise ValueError('Explicit current formal transition registration required')
    elif row.get('phase')!='development' or args.formal:
        raise ValueError('Development entry cannot imply formal authorization')


def main(args):
    from research.v4.development.confidence_gate_dev_20261006 import native, runner, backend, evidence, runtime
    from research.v4.development.confidence_gate_dev_20261006 import common_control
    from research.v4.development.confidence_gates_20261006 import config
    from research.v4.leash_v43 import run_manifest as manifests
    plan=json.loads(args.plan.read_text())
    row=next(r for r in plan['episodes'] if r['run_id']==args.output.name)
    validate_registration(args,plan,row)
    from research.v4.development.score_only_g3_030_20261021 import profile as score_profile
    score_only = score_profile.enabled(plan)
    if score_only:
        score_profile.validate(plan, row, args)
    original_adapter=runner.ConfidenceAdapter
    original_intent=common_control.world_reference_intent
    original_manifest=manifests.run_manifest
    original_stack=runner.create_episode_stack

    class TransitionAdapter(original_adapter):
        def __init__(self,*values,**kwargs):
            with patch.object(runtime,'HoldingEstimator',CompletedOpenHoldingEstimator):
                super().__init__(*values,**kwargs)
            old=self.dispatcher
            self.registry=TransitionRegistry(self.estimator,
                position_tolerance_m=self.registry.position_tolerance_m,
                direction_tolerance_rad=self.registry.direction_tolerance_rad,
                method='Ours',require_grasp_assumption=False,semantic_unknown_pour=True)
            # Keep the original exact dispatcher type for the existing owner
            # cancellation binder. Only its instance-local chunk staging is
            # replaced; step/abort/send and their identity checks stay intact.
            self.dispatcher=type(old)(registry=self.registry,decode=old.decode,
                read_proprio=old.read_proprio,send=old.send,capture=old.capture,
                review=self._review,project=self._project,method='Ours',enabled=True,
                review_timeout_s=old.review_timeout_s)
            self.dispatcher.sequence=old.sequence
            self.dispatcher.run_chunk=MethodType(TransitionDispatcher.run_chunk,self.dispatcher)

        def _project(self,**kwargs):
            request=super()._project(**kwargs)
            request['evidence']['reviewer_prompt_version']=score_profile.VERSION if score_only else VERSION
            proposal=self.registry.pending_window
            if proposal is not None:
                scope=json.loads(json.dumps(proposal.descriptor))
                request['evidence']['event']['continuation']['reviewed_window']=scope
                scope_hash=digest(scope)
                request['candidate']['transition_scope_sha256']=scope_hash
                request['evidence']['candidate']['transition_scope_sha256']=scope_hash
                request['evidence']['review_scope']=dict(
                    kind='pending_commit_with_explicit_finite_continuation',
                    scored_field='candidate.control + event + event.continuation.reviewed_window',
                    context_only_fields=['trusted_instruction','parent_primitive_context'],
                    excluded_scope='all unlisted controls, later VLA chunks, new gripper phases and later primitives')
            return request

    @contextmanager
    def intent(dispatcher,description,trace):
        with prepared_motion_scope(original_intent,dispatcher,description,trace):yield

    @contextmanager
    def recorded(*values,**kwargs):
        kwargs['source_files']=list(dict.fromkeys(list(kwargs['source_files'])+
            [str(p.relative_to(ROOT)) for p in PACKAGE.glob('*') if p.suffix in ('.py','.txt')]+
            ['research/v4/development/g3_event_count_repair_20261020/'+n
             for n in ('__init__.py','estimator.py','registry.py')]+
            (['research/v4/bridge/formal_optimized_matrix.py'] if plan.get('optimized_matrix') else [])))
        if score_only:
            kwargs['source_files'] += [str(p.relative_to(ROOT)) for p in
                (ROOT/'research/v4/development/score_only_g3_030_20261021').glob('*.py')]
        with original_manifest(*values,**kwargs) as data:
            data['g3_transition_development']=dict(version=VERSION,formal=bool(args.formal),
                planner=row.get('planner_model','gpt-6-astra')+'/high',
                reviewer={'luna':'gpt-6-luna/high','openai_decisions':'gpt-6-luna/predicate','jev':'jev-1.14.0/noul'}[args.reviewer_backend],
                G1_G2_policy_examples_physics_scoring_unchanged=True)
            data['g3_event_count_repair']=dict(version=EVENT_COUNT_REPAIR_VERSION,
                exact_source_sequence_index_and_gripper_binding=True,
                completed_open_retained_only_while_current_public_evidence_agrees=True,
                thresholds_policy_common_control_physics_scoring_unchanged=True)
            if score_only:
                data['reviewer_output_contract'] = plan['reviewer_output_contract']
                data['g3_event_count_repair']['thresholds_policy_common_control_physics_scoring_unchanged'] = False
                data['g3_event_count_repair']['authorized_thresholds'] = score_profile.THRESHOLDS
                data['g3_event_count_repair']['policy_control_physics_scoring_unchanged'] = True
            if plan.get('private_outcome_review'):
                data['private_outcome_review'] = plan['private_outcome_review']
                data['g3_transition_development']['G1_G2_policy_examples_physics_scoring_unchanged'] = False
                data['g3_transition_development']['G1_G2_policy_examples_physics_unchanged'] = True
                data['g3_event_count_repair']['policy_control_physics_scoring_unchanged'] = False
                data['g3_event_count_repair']['policy_control_physics_unchanged'] = True
            if row.get('policy_revision'):
                data['shared_policy_revision'] = row['policy_revision']
                data['g3_transition_development'].update(
                    G1_G2_policy_examples_physics_scoring_unchanged=False,
                    G1_G2_policy_examples_physics_unchanged=False,
                    examples_physics_scoring_unchanged=True)
                data['g3_event_count_repair'].update(
                    thresholds_policy_common_control_physics_scoring_unchanged=False,
                    policy_control_physics_scoring_unchanged=False,
                    policy_control_physics_unchanged=False,
                    thresholds_common_control_physics_scoring_unchanged=True)
            yield data

    @contextmanager
    def stack_scope(**kwargs):
        with original_stack(**kwargs) as stack:
            try:yield stack
            finally:
                (args.output/'transition_windows.json').write_text(json.dumps(
                    stack.adapter.registry.window_logs,ensure_ascii=False,indent=2)+'\n')

    with ExitStack() as changes:
        changes.enter_context(patch.object(runner,'ConfidenceAdapter',TransitionAdapter))
        changes.enter_context(patch.object(runner,'create_episode_stack',stack_scope))
        changes.enter_context(patch.object(common_control,'world_reference_intent',intent))
        if score_only:
            from research.v4.development.score_only_g3_030_20261021.reviewer import ScoreOnlyLunaWorker, ScoreOnlySession
            luna_worker, score_session = ScoreOnlyLunaWorker, ScoreOnlySession
        else:
            luna_worker, score_session = TransitionLunaWorker, TransitionScoreSession
        changes.enter_context(patch.object(backend,'LunaWorker',luna_worker))
        changes.enter_context(patch.object(backend,'ScoreSession',score_session))
        if matrix := plan.get('optimized_matrix'):
            # Inject the identical finite G3 question into typed endpoints;
            # observation projection, G1/G2, policy and thresholds stay shared.
            from research.v4.development.decision_gate_dev_20261019 import backend as typed
            from .reviewer import scoped_prompt
            reviewer_prompt = score_profile.score_only_prompt if score_only else scoped_prompt
            reviewer_version = score_profile.VERSION if score_only else VERSION
            original_decision,original_jev=typed.DecisionBackend,typed.JevBackend
            def scoped_factory(cls):
                def create(*values,**kwargs):
                    return cls(*values,**kwargs,prompt_provider=reviewer_prompt,prompt_version=reviewer_version)
                return create
            changes.enter_context(patch.object(typed,'DecisionBackend',scoped_factory(original_decision)))
            changes.enter_context(patch.object(typed,'JevBackend',scoped_factory(original_jev)))
        changes.enter_context(patch.object(manifests,'run_manifest',recorded))
        return native.main(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('Ours-conf',),required=True)
    parser.add_argument('--instance',required=True);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--gpu',type=int,default=1);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--planner-model',default='gpt-6-astra')
    parser.add_argument('--reviewer-backend',default='luna')
    parser.add_argument('--formal',action='store_true')
    for gate in ('task','action','commit'):parser.add_argument('--tau-'+gate,type=float,default=.5)
    raise SystemExit(main(parser.parse_args()))
