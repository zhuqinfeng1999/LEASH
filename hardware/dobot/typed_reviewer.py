"""Select the frozen typed reviewer using process-local environment credentials."""
import os
from pathlib import Path

from .reference_text import make_backend

REVIEWER = {
    "backend": "openai_decisions",
    "model": "gpt-6-luna",
    "question_type": "predicate",
    "score_visible_to_planner": False,
}
SOURCE_FREEZE = "research/v4/handoff_20261020/simulation_source_freeze.json"
REVIEWERS = {
    'openai_decisions': REVIEWER,
    'jev': dict(backend='jev', model='jev-1.14.0', question_type='noul',
                score_visible_to_planner=False),
}


def reviewer_name(config):
    selected = config.get('reviewer')
    if not isinstance(selected, dict) or selected != REVIEWERS.get(selected.get('backend')):
        raise ValueError('Explicit pinned Decisions predicate or Jev Noul configuration required')
    return selected['backend']


def private_key(backend):
    if backend not in REVIEWERS:
        raise ValueError('Unknown hardware reviewer credential')
    env_name = 'OPENAI_API_KEY' if backend == 'openai_decisions' else 'TYPESAFE_API_KEY'
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"Set {env_name} for the selected reviewer")
    return value


class DisabledReviewer:
    name = "disabled_B0"

    def score(self, *_):
        raise RuntimeError("B0 must never call a safety reviewer")

    def close(self):
        pass


def create_reviewer(config, output, *, method):
    name = reviewer_name(config)
    if method == "B0":
        return DisabledReviewer()
    if method != "Ours":
        raise ValueError("Unknown experiment method")
    backend = make_backend(name, output, key_provider=private_key, timeout_s=180.)
    if config.get('hardware_policy_revision'):
        from .hardware_policy import effective_policy, install_backend_policy
        from research.v4.development.confidence_gates_20261006.config import policy
        text, provenance = effective_policy(config, policy())
        backend = install_backend_policy(backend, text, provenance)
    return backend
