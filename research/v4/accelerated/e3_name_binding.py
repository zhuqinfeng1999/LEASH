"""Map independent reviewer entity names to E3's public RGB-D labels.

Only explicit names are accepted. The bindings expose no simulator identity,
grasp zone, task truth, or offline evaluator geometry.
"""
from __future__ import annotations

import re
from dataclasses import replace


PROMPTS = {
    "kettle": "small grey kettle",
    "mug": "blue mug",
    "person": "seated person",
    "hand": "human hand",
}


def canonical_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("reviewer entity name must be text")
    name = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    aliases = {
        "kettle": "kettle", "hot water kettle": "kettle",
        "water kettle": "kettle", "hot water vessel": "kettle",
        "pourer": "kettle", "teapot": "kettle",
        "mug": "mug", "blue mug": "mug", "cup": "mug",
        "target mug": "mug", "target cup": "mug",
        "person": "person", "seated person": "person",
        "human": "person", "human body": "person",
        "hand": "hand", "human hand": "hand",
        "right hand": "hand", "left hand": "hand",
        "person s hand": "hand", "person hand": "hand",
        "person s right hand": "hand", "person s left hand": "hand",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(f"unbound E3 public entity name: {value!r}") from exc


def bind_task_names(decision):
    if decision.result != "PROCEED":
        return decision, []
    records = []

    def bind(values, field):
        answer = set()
        for original in sorted(values):
            bound = canonical_name(original)
            records.append({"field": field, "original": original, "bound": bound})
            answer.add(bound)
        return frozenset(answer)

    old = decision.constraints
    allowed = bind(old.allowed_operated_entities, "allowed_operated_entities")
    forbidden = bind(old.forbidden_entities, "forbidden_entities")
    rules = []
    for rule in old.orientation_rules:
        bound = canonical_name(rule.operated_entity)
        records.append({"field": "orientation_rules", "original": rule.operated_entity,
                        "bound": bound})
        rules.append(replace(rule, operated_entity=bound))
    if "kettle" not in allowed:
        raise ValueError("task reviewer did not allow the user-requested kettle")
    return replace(decision, constraints=replace(
        old, allowed_operated_entities=allowed,
        forbidden_entities=forbidden, orientation_rules=tuple(rules))), records


def bind_motion_names(raw):
    values = raw.get("forbidden_entities", [])
    if not isinstance(values, list):
        raise ValueError("motion reviewer forbidden_entities is not a list")
    records = []
    mapped = []
    for original in values:
        bound = canonical_name(original)
        records.append({"field": "forbidden_entities", "original": original,
                        "bound": bound})
        mapped.append(bound)
    result = dict(raw)
    result["forbidden_entities"] = sorted(set(mapped))
    return result, records
