"""Neutral development holdout parameters over the existing collector.

This is not a runner or scheduler. Root owns all starts, services, manifests,
limits, output paths and evaluation. Holdout outcomes cannot choose examples.
"""
from dataclasses import dataclass

from research.v4.familiarization.collector_draft import entry as collector
from .binding import wrap_planner_builder


HOLDOUT_SEEDS = {
    "box": (750100, 750101, 750102),
    "cylinder": (750200, 750201, 750202),
    "empty_handled_container": (750300, 750301, 750302),
    "neutral_flat_object": (750400, 750401, 750402),
}
PREREGISTERED_FAMILY = {"box": "cube_or_box", "cylinder": "cylinder",
                       "empty_handled_container": "empty_handled_container",
                       "neutral_flat_object": "neutral_flat_object"}
CONDITIONS = ("no_familiarization", "shared_familiarization")


@dataclass(frozen=True)
class HoldoutPlan:
    family: str
    seed: int
    condition: str = "no_familiarization"
    model_id: str = "gpt-5.5"
    reasoning_effort: str = "high"
    planner_timeout_s: int = 1500
    max_turns: int = 80
    primitive_calls_per_round: int = 30
    primitive_calls_per_episode: int = 90
    episode_wall_seconds: int = 1800
    max_output_bytes: int = 1024 ** 3
    phase: str = "development_holdout"
    used_for_example_selection: bool = False
    state_mapping: str = "restored_original_libero_axisangle"
    frozen_vla: bool = True

    def validate(self):
        if self.family not in HOLDOUT_SEEDS:
            raise ValueError("Only the four preregistered neutral holdout families are allowed")
        if type(self.seed) is not int or self.seed not in HOLDOUT_SEEDS[self.family]:
            raise ValueError("Use only the matching preregistered 750xxx development holdout seed")
        if self.condition not in CONDITIONS:
            raise ValueError("Use the two preregistered familiarization conditions")
        if (self.model_id, self.reasoning_effort) != ("gpt-5.5", "high"):
            raise ValueError("Keep the same GPT-5.5/high planner")
        fixed = ((self.planner_timeout_s, 1500), (self.max_turns, 80),
                 (self.primitive_calls_per_round, 30), (self.primitive_calls_per_episode, 90),
                 (self.episode_wall_seconds, 1800), (self.max_output_bytes, 1024 ** 3))
        if any(type(actual) is not int or actual != expected for actual, expected in fixed):
            raise ValueError("Keep the existing finite development budgets")
        if self.phase != "development_holdout" or self.used_for_example_selection is not False:
            raise ValueError("Holdout is development-only and cannot select/update examples or become formal")
        if self.state_mapping != "restored_original_libero_axisangle" or self.frozen_vla is not True:
            raise ValueError("Keep restored current input mapping and frozen pi0.5")
        return self

    @property
    def pair_id(self):
        self.validate()
        layout = HOLDOUT_SEEDS[self.family].index(self.seed) + 1
        return f"HOLDOUT-{PREREGISTERED_FAMILY[self.family].upper()}-{layout:02d}"

    def manifest_fields(self):
        self.validate()
        return dict(phase=self.phase, role="neutral_development_holdout", formal=False,
            family=self.family, seed=self.seed, pair_id=self.pair_id, condition=self.condition,
            used_for_example_selection=False, planner_model=self.model_id,
            planner_reasoning=self.reasoning_effort, method="B0", gates=[],
            reviewer_model=None, reviewer_reasoning=None,
            state_mapping=self.state_mapping, frozen_vla=True,
            budgets=dict(primitive_calls_per_round=self.primitive_calls_per_round,
                         primitive_calls_per_episode=self.primitive_calls_per_episode,
                         episode_wall_seconds=self.episode_wall_seconds,
                         max_output_bytes=self.max_output_bytes),
            public_input="Same existing agentview/wrist RGB-D, proprioception, SAM3 and pi0.5 preprocessing",
            global_memory="Same Root-selected existing Global Memory; no persistent writes")


def interleaved_plans():
    """Return the already preregistered 12 pairs/24 condition order, execute none."""
    plans, pair = [], 0
    for family, seeds in HOLDOUT_SEEDS.items():
        for seed in seeds:
            pair += 1
            conditions = CONDITIONS if pair % 2 else tuple(reversed(CONDITIONS))
            plans.extend(HoldoutPlan(family, seed, condition).validate() for condition in conditions)
    return tuple(plans)


def build_context(plan, **kwargs):
    """Delegate to current collector; Root alone supplies real clients/Scene start."""
    plan.validate()
    return collector.build_context(plan, **kwargs)


def run_holdout_round(context, plan, binding, *, planner_builder=None):
    """Existing collector/Owner loop with only the declared prompt binding.

    No layout/outcome, private state or completion metric is read to select or
    modify example content. Root freezes binding before examining holdout data.
    The existing loop enforces B0/no custody/fresh round/30-and-90 call budgets.
    """
    plan.validate()
    if binding.condition != plan.condition:
        raise ValueError("Holdout condition and frozen prompt binding must agree")
    if planner_builder is None:
        from research.v3.bridge.b0 import build_b0_planner
        planner_builder = build_b0_planner
    selected_builder = wrap_planner_builder(planner_builder, binding)
    return collector.run_collection_round(context, plan, planner_builder=selected_builder)
