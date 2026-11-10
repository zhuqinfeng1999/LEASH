"""Bind free-form reviewer entity names to E1 public RGB-D labels.

No simulator name, true pose, or evaluator flag is used. Unknown/compound
names fail closed; original model text and every mapping are retained.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping

from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    TaskGateDecision,
)


PROMPTS = {
    "plate": "blue plate",
    "knife": "red handle",
    "person": "seated person",
    "hand": "human hand",
}


def canonical_name(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    aliases = {
        "plate": "plate", "blue plate": "plate", "dish": "plate",
        "target plate": "plate", "knife": "knife", "blade": "knife",
        "knife blade": "knife", "knife handle": "knife",
        "hand": "hand", "human hand": "hand", "person s hand": "hand",
        "person hand": "hand", "left hand": "hand", "right hand": "hand",
        "person s left hand": "hand", "person s right hand": "hand",
        "human": "person", "person": "person", "seated person": "person",
        "human body": "person", "person s body": "person",
    }
    if text not in aliases:
        raise ValueError(f"unbound E1 public entity name: {name!r}")
    return aliases[text]


def bind_task_names(decision: TaskGateDecision) -> tuple[TaskGateDecision, list[dict[str, str]]]:
    """Normalize only exact known aliases; do not drop constraints."""
    if decision.result != "PROCEED":
        return decision, []
    records: list[dict[str, str]] = []

    def names(values: frozenset[str], field: str) -> frozenset[str]:
        mapped = set()
        for original in sorted(values):
            bound = canonical_name(original)
            records.append({"field": field, "original": original, "bound": bound})
            mapped.add(bound)
        return frozenset(mapped)

    constraints = decision.constraints
    allowed = names(constraints.allowed_operated_entities, "allowed_operated_entities")
    forbidden = names(constraints.forbidden_entities, "forbidden_entities")
    rules = []
    for rule in constraints.orientation_rules:
        bound = canonical_name(rule.operated_entity)
        records.append({"field": "orientation_rules", "original": rule.operated_entity,
                        "bound": bound})
        rules.append(replace(rule, operated_entity=bound))
    if "knife" not in allowed:
        raise ValueError("task reviewer did not allow the user-requested knife")
    return replace(decision, constraints=replace(
        constraints, allowed_operated_entities=allowed,
        forbidden_entities=forbidden, orientation_rules=tuple(rules))), records


def bind_motion_names(raw: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Normalize gate-2 additions, leaving every other reviewer field intact."""
    values = raw.get("forbidden_entities", [])
    if not isinstance(values, list):
        raise ValueError("motion reviewer forbidden_entities is not a list")
    records, mapped = [], []
    for original in values:
        if not isinstance(original, str):
            raise ValueError("motion reviewer entity name is not text")
        bound = canonical_name(original)
        records.append({"field": "forbidden_entities", "original": original,
                        "bound": bound})
        mapped.append(bound)
    result = dict(raw)
    result["forbidden_entities"] = sorted(set(mapped))
    return result, records
