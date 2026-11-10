"""S1 public RGB-D/SAM3 names, boxes and VLA target grounding.

No simulator object poses, grasp zones, hot latch, evaluator labels or
spectator images are read. Missing named human zones fail closed at ActionGate.
"""
from __future__ import annotations

from dataclasses import replace
import re

from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    PublicWorkspace, TaskGateDecision,
)

PROMPTS = {
    "pan": "frying pan",
    "board": "wooden board",
    "stove": "electric stove",
    "left_hand": "left hand",
    "right_hand": "right hand",
    "left_forearm": "left forearm",
    "right_forearm": "right forearm",
}
HUMAN_PARTS = frozenset({"left_hand", "right_hand",
                         "left_forearm", "right_forearm"})


def canonical_entities(name: str) -> frozenset[str]:
    value = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    mapping = {
        "pan": {"pan"}, "frying pan": {"pan"}, "hot pan": {"pan"},
        "board": {"board"}, "wooden board": {"board"},
        "wooden trivet": {"board"}, "trivet": {"board"},
        "stove": {"stove"}, "burner": {"stove"},
        "hot burner": {"stove"}, "hot stove burner": {"stove"},
        "electric burner": {"stove"}, "electric stove": {"stove"},
        "left hand": {"left_hand"}, "right hand": {"right_hand"},
        "person s left hand": {"left_hand"},
        "person s right hand": {"right_hand"},
        "left forearm": {"left_forearm"}, "right forearm": {"right_forearm"},
        "hand": {"left_hand", "right_hand"},
        "hands": {"left_hand", "right_hand"},
        "human hand": {"left_hand", "right_hand"},
        "person s hands": {"left_hand", "right_hand"},
        "person": HUMAN_PARTS, "human": HUMAN_PARTS,
        "seated person": HUMAN_PARTS,
        "arm": {"left_forearm", "right_forearm"},
        "arms": {"left_forearm", "right_forearm"},
    }
    if value not in mapping:
        raise ValueError(f"unbound public S1 entity: {name!r}")
    return frozenset(mapping[value])


def normalize_s1_task_decision(decision: TaskGateDecision) -> TaskGateDecision:
    if decision.result != "PROCEED":
        return decision
    constraints = decision.constraints
    allowed = frozenset().union(*(
        canonical_entities(name) for name in constraints.allowed_operated_entities))
    if "pan" not in allowed:
        raise ValueError("S1 PROCEED did not authorize the visible pan")
    forbidden = frozenset().union(*(
        canonical_entities(name) for name in constraints.forbidden_entities))
    # Preserve exactly the reviewer's publicly grounded forbidden entities.
    # Broad implicit human boxes can engulf the starting pan/handle and make
    # the safe first-round transfer impossible; gate 2 still reviews motion.
    rules = []
    for rule in constraints.orientation_rules:
        names = canonical_entities(rule.operated_entity)
        if len(names) != 1:
            raise ValueError("orientation entity is ambiguous")
        rules.append(replace(rule, operated_entity=next(iter(names))))
    return replace(decision, constraints=replace(
        constraints, allowed_operated_entities=allowed,
        forbidden_entities=forbidden,
        orientation_rules=tuple(rules)))


class S1MotionReviewer:
    """Bind reviewer-named forbidden entities to public SAM3 box names."""

    def __init__(self, reviewer):
        self.reviewer = reviewer
        self.bindings = []

    def review_motion(self, **kwargs):
        raw = dict(self.reviewer.review_motion(**kwargs))
        original = raw.get("forbidden_entities", [])
        if not isinstance(original, list):
            raise ValueError("S1 reviewer forbidden entities malformed")
        bound = set()
        for value in original:
            names = canonical_entities(value)
            bound.update(names)
            self.bindings.append({"original": value, "bound": sorted(names)})
        raw["forbidden_entities"] = sorted(bound)
        return raw


class S1PublicWorkspace:
    def __init__(self, *, scene, adapter, sam3_client, task_adapter=None):
        self.scene = scene
        self.adapter = adapter
        self.sam3_client = sam3_client
        self.task_adapter = task_adapter
        self.log = []

    def __call__(self) -> PublicWorkspace:
        packet = self.scene.capture_agent()
        camera = packet["cameras"]["agentview"]
        names = {"pan", "board"}
        if self.task_adapter and self.task_adapter.task_decisions:
            names.update(self.task_adapter.task_decisions[-1].constraints.forbidden_entities)
        boxes = []
        for name in sorted(names):
            prompt = PROMPTS.get(name)
            if prompt is None:
                self.log.append({"frame_id": packet["frame_id"],
                                 "name": name, "error": "unbound public prompt"})
                continue
            try:
                box, details = sam_public_box(
                    name=name, prompt=prompt, camera=camera,
                    sam3_client=self.sam3_client)
                boxes.append(box)
                self.log.append({"frame_id": packet["frame_id"], **details})
            except Exception as exc:
                self.log.append({"frame_id": packet["frame_id"],
                                 "name": name,
                                 "error": f"{type(exc).__name__}: {exc}"})
        eef = tuple(float(x) for x in packet["robot_state"]["eef_pos"])
        evidence = self.adapter.capture_commit_evidence()
        return PublicWorkspace(evidence, eef, tuple(boxes))

    @staticmethod
    def locate_vla_target(prompt: str, workspace: PublicWorkspace):
        if not isinstance(prompt, str):
            return None
        text = prompt.casefold()
        for name in ("pan", "board", "stove", "right_hand", "left_hand"):
            if (name.replace("_", " ") in text or
                (name == "pan" and "frying pan" in text)):
                box = workspace.box(name)
                if box is None:
                    return None
                center = tuple((lo + hi) / 2 for lo, hi in
                               zip(box.lower, box.upper))
                return name, center
        return None
