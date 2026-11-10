"""S1 reviewer phrase aliases, grounded only to public camera parts.

The reviewer emits natural phrases. This layer changes only name spelling; it
never reads scene objects, collision geoms, evaluator truth or hidden poses.
The existing public SAM3 RGB-D prompts must still locate each named part.
"""
from __future__ import annotations

import re

from research.v4.candidates.s1_multiround_driver_20260925 import public_grounding

_ORIGINAL = public_grounding.canonical_entities
_HANDS = frozenset({"left_hand", "right_hand"})
_FOREARMS = frozenset({"left_forearm", "right_forearm"})
_VISIBLE_PERSON_PARTS = public_grounding.HUMAN_PARTS


def canonical_entities_v3(name: str) -> frozenset[str]:
    if not isinstance(name, str):
        raise ValueError(f"unbound public S1 entity: {name!r}")
    phrase = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    if phrase in {"person hands", "person s hands"}:
        return _HANDS
    if phrase in {"person body", "person s body", "person s head",
                  "person s torso"}:
        # A deliberately conservative proxy for the reachable, publicly
        # segmented human parts; never claim a hidden head/torso truth box.
        return _VISIBLE_PERSON_PARTS
    if phrase == "person s arms except open receiving hands":
        return _FOREARMS
    if phrase == "table edge":
        return frozenset({"table_edge"})
    return _ORIGINAL(name)


def install_public_name_aliases() -> None:
    current = public_grounding.canonical_entities
    if current is canonical_entities_v3:
        return
    if current is not _ORIGINAL:
        raise RuntimeError("unexpected S1 canonicalizer; refuse to overwrite")
    # Existing task-decision and motion-review functions resolve this global
    # at call time. Method runner imported those functions, not this global.
    public_grounding.PROMPTS["table_edge"] = "visible front edge of the table"
    public_grounding.canonical_entities = canonical_entities_v3
