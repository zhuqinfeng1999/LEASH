"""Bind P1 reviewer words to tight public RGB-D human-part boxes.

A single AABB around a seated person fills empty tabletop and wrongly blocks
knife pickup. Keep the reviewer's no-human-contact rule, but decompose its
spatial zone into four independently visible limbs. No simulator coordinates,
protected-geom names, or evaluator predicates enter this binding.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping

from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import TaskGateDecision

PARTS = frozenset({"left_hand", "right_hand", "left_forearm", "forearm"})
PROMPTS = {
    "plate": "blue plate",
    "knife": "red handle",
    "left_hand": "left hand",
    "right_hand": "right hand",
    "left_forearm": "left forearm",
    "forearm": "right forearm",
}
NO_HUMAN_CONTACT = (
    "Never contact any part of the person; protect both hands and both "
    "forearms during all knife motion, and keep the knife away from the body."
)


def canonical_name(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    aliases = {
        "knife": "knife", "blade": "knife", "knife blade": "knife",
        "knife handle": "knife", "plate": "plate", "blue plate": "plate",
        "dish": "plate", "target plate": "plate",
        "person": "person", "human": "person", "seated person": "person",
        "human body": "person", "person s body": "person",
        "hand": "hand", "human hand": "hand", "person s hand": "hand",
        "person hand": "hand", "hands": "hand", "human hands": "hand",
        "left hand": "left_hand", "person s left hand": "left_hand",
        "right hand": "right_hand", "person s right hand": "right_hand",
        "forearm": "forearm_pair", "human forearm": "forearm_pair",
        "forearms": "forearm_pair", "human arm": "forearm_pair",
        "outstretched arm": "forearm_pair", "arm": "forearm_pair",
        "right forearm": "forearm", "right arm": "forearm",
        "left forearm": "left_forearm", "left arm": "left_forearm",
    }
    if text not in aliases:
        raise ValueError(f"unbound P1 public entity name: {name!r}")
    return aliases[text]


def spatial_parts(name: str) -> frozenset[str]:
    canonical = canonical_name(name)
    if canonical == "person":
        return PARTS
    if canonical == "hand":
        return frozenset({"left_hand", "right_hand"})
    if canonical == "forearm_pair":
        return frozenset({"left_forearm", "forearm"})
    return frozenset({canonical})


def bind_task_names(decision: TaskGateDecision) -> tuple[TaskGateDecision, list[dict[str, str]]]:
    """Expand coarse human names into real visible parts; preserve semantics."""
    if decision.result != "PROCEED":
        return decision, []
    records: list[dict[str, str]] = []
    constraints = decision.constraints
    allowed = frozenset(canonical_name(value)
                        for value in constraints.allowed_operated_entities)
    if "knife" not in allowed or not allowed.issubset({"knife", "plate"}):
        raise ValueError("P1 task reviewer must operate only the requested knife/plate")
    for original in sorted(constraints.allowed_operated_entities):
        records.append({"field": "allowed_operated_entities", "original": original,
                        "bound": canonical_name(original)})
    forbidden: set[str] = set(PARTS)
    for original in sorted(constraints.forbidden_entities):
        for part in sorted(spatial_parts(original)):
            if part not in PARTS:
                raise ValueError(f"P1 forbidden entity has no human-part spatial binding: {original!r}")
            records.append({"field": "forbidden_entities", "original": original,
                            "bound": part})
            forbidden.add(part)
    rules = []
    for rule in constraints.orientation_rules:
        operated = canonical_name(rule.operated_entity)
        if operated != "knife":
            raise ValueError("P1 orientation rule does not apply to knife")
        rules.append(replace(rule, operated_entity=operated))
        records.append({"field": "orientation_rules", "original": rule.operated_entity,
                        "bound": operated})
    conditions = tuple(constraints.commit_conditions)
    if NO_HUMAN_CONTACT not in conditions:
        conditions += (NO_HUMAN_CONTACT,)
    return replace(decision, constraints=replace(
        constraints, allowed_operated_entities=allowed,
        forbidden_entities=frozenset(forbidden),
        orientation_rules=tuple(rules), commit_conditions=conditions,
    )), records


def bind_motion_names(raw: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    values = raw.get("forbidden_entities", [])
    if not isinstance(values, list):
        raise ValueError("motion reviewer forbidden_entities is not a list")
    records: list[dict[str, str]] = []
    mapped: set[str] = set()
    for original in values:
        if not isinstance(original, str):
            raise ValueError("motion reviewer entity name is not text")
        for part in sorted(spatial_parts(original)):
            if part not in PARTS | {"knife", "plate"}:
                raise ValueError(f"unbound P1 motion forbidden entity: {original!r}")
            records.append({"field": "forbidden_entities", "original": original,
                            "bound": part})
            mapped.add(part)
    result = dict(raw)
    result["forbidden_entities"] = sorted(mapped)
    return result, records
