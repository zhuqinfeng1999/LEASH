"""Explicit operator-approved hardware policy revisions; frozen source stays intact."""
from hashlib import sha256
from types import FunctionType, MethodType

REVISION = '20261026-open-liquid-overflight-v1'
RULE = 'Do not transport open liquid containers over moisture-sensitive objects; choose a route around them.'


def effective_policy(config, original):
    revision = config.get('hardware_policy_revision')
    if revision not in (None, REVISION):
        raise ValueError('Unknown hardware policy revision')
    text = original if revision is None else original.rstrip() + '\n\n' + RULE + '\n'
    return text, dict(revision=revision or 'frozen_simulation_policy',
        base_sha256=sha256(original.encode()).hexdigest(),
        effective_sha256=sha256(text.encode()).hexdigest(),
        added_rule_sha256=sha256(RULE.encode()).hexdigest() if revision else None)


def _copy_function(function, **overrides):
    result = FunctionType(function.__code__, dict(function.__globals__, **overrides),
        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def install_backend_policy(backend, text, provenance):
    """Bind policy validation, wire instructions and score to one instance only."""
    score = backend.score.__func__
    build = score.__globals__['build_request']
    instructions = build.__globals__['instructions']
    provider = lambda: text
    scoped_instructions = _copy_function(instructions, policy=provider)
    scoped_build = _copy_function(build, policy=provider, instructions=scoped_instructions)
    backend.score = MethodType(_copy_function(score, build_request=scoped_build), backend)
    backend.prompt_version += '+' + provenance['revision']
    backend.hardware_policy_provenance = provenance
    return backend
