"""Small bindings for the existing confidence CLI's non-P102 entry.

Import and configuration lookup do not start scenes or services. Private
evaluation remains a runner sink. The original S driver still owns its
session, finish/quota boundaries and next-round decision.
"""
from contextlib import contextmanager
import os

from .configurations import configuration


def select_configuration(instance):
    if instance == 'P1-03':
        raise ValueError('P1-03 grasp-chain acceptance is pending; do not launch')
    if instance in ('P1-01','P1-02'):
        from .runner import p1_configuration
        return p1_configuration(instance)
    return configuration(instance)


def make_private_evaluator(configured, scene):
    """Construct once in on_scene, before controls; no value enters gates."""
    return (configured.Evaluator(scene) if getattr(configured, 'evaluator_requires_scene', False)
            else configured.Evaluator())


def finish_private_evaluator(configured, evaluator, terminal, *, final_scene=None):
    # S2/S3 obtain their actual per-round terminal through existing hooks or
    # terminal_provider. Their accepted scorer's finish takes no status arg.
    family = configured.binding.instance.split('-')[0]
    if family in ('S2', 'S3'):
        return evaluator.finish()
    status = None if terminal is None else terminal.get('status')
    if status not in ('done', 'asked', 'refused'):
        status = None
    if getattr(configured, 'evaluator_requires_final_scene', False):
        return evaluator.finish(status, final_scene=final_scene)
    return evaluator.finish(status)


@contextmanager
def persistent_budget_scope(stack):
    """Select the already-existing 60/180/2700 S budget branch locally."""
    from research.v4.bridge.protocol import active_live_limits
    if os.environ.get('SAFEVLA_EXECUTION_PROFILE') not in ('operation_development_20261004', 'user_review_development_20261008', 'confidence_formal_20261008'):
        raise ValueError('S confidence entry requires the registered current resource profile')
    limits = active_live_limits()
    if (stack.base.per_round_budget, stack.base.episode_budget) != (
            limits['primitive_calls_per_round'], limits['primitive_calls_per_episode']):
        raise ValueError('S stack and existing driver resource profile disagree')
    old = os.environ.get('SAFEVLA_LEASH_BACKEND')
    os.environ['SAFEVLA_LEASH_BACKEND'] = 'live'
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('SAFEVLA_LEASH_BACKEND', None)
        else:
            os.environ['SAFEVLA_LEASH_BACKEND'] = old


def prepare_persistent_entry(stack, configured, *, evaluator, output_dir,
                             model_id, reasoning_effort='high', prompt_binding=None,
                             episode_started_at=None, holding_reader=None,
                             session_factory=None, coordinator_factory=None, on_round_prepared=None):
    """Return scope/body/hooks for Root's existing actual Owner pump.

    Caller constructs this after create_episode_stack and on_scene evaluator
    setup. Call body via stack.owner.run_planner, preserving that same stack.
    The returned hooks expose seals/recovery for ordinary output serialization.
    """
    from .persistent import make_persistent_callback
    from .s_round_hooks import make_hooks
    from .s1_recovery import make_recovery_executor
    if not configured.persistent:
        raise ValueError('Single-round configuration uses run_single_episode')
    if on_round_prepared is not None and not callable(on_round_prepared):
        raise TypeError('on_round_prepared must be an owner-side callable or None')
    hooks = make_hooks(stack, configured, evaluator=evaluator,
                       holding_reader=holding_reader)
    if configured.binding.instance.startswith('S1-'):
        hooks.recovery_executor = make_recovery_executor(
            stack, seal_reader=lambda: hooks.seals[-1]['seal'])
    def prepare_physical_round(index, instruction, previous):
        result = hooks.prepare_physical_round(index, instruction, previous)
        if on_round_prepared is not None:
            # The accepted evaluator has now started this actual round. Root
            # may save its private row before G1 without inventing a step.
            on_round_prepared(index, instruction, previous)
        return result

    options = dict(output_dir=output_dir, model_id=model_id,
        reasoning_effort=reasoning_effort,
        prepare_physical_round=prepare_physical_round,
        after_round=hooks.after_round, prompt_binding=prompt_binding,
        episode_started_at=episode_started_at)
    if session_factory is not None:
        options['session_factory'] = session_factory
    if coordinator_factory is not None:
        options['coordinator_factory'] = coordinator_factory
    round_scope, body = make_persistent_callback(stack, configured, **options)

    @contextmanager
    def scope():
        from research.v4.development.g2_candidate_evidence_dev_20261005.capacity_runtime import enabled, owner_scope
        with persistent_budget_scope(stack), round_scope:
            if enabled():
                with owner_scope(stack.owner):
                    yield hooks
            else:
                yield hooks
    return scope(), body, hooks


def cli_readiness(instance):
    """Actual config capability, not an assertion of natural-run success."""
    configured = select_configuration(instance)
    instructions = getattr(configured, 'instructions', (configured.binding.instruction,))
    return dict(instance=instance, instructions=list(instructions),
        persistent=bool(getattr(configured, 'persistent', False)),
        physical_binding_ready=True, native_confidence_verified=False,
        empty_reset_qualified=bool(getattr(configured, 'empty_reset_qualified', False)),
        cli_currently_enabled=instance == 'P1-02',
        evaluator_requires_scene=bool(getattr(configured, 'evaluator_requires_scene', False)),
        entry='existing_persistent_driver' if getattr(configured, 'persistent', False)
              else 'existing_single_episode_runner')
