"""P3-only normalization of live reviewer hazard names.

A protected laptop zone requires two agreeing public SAM prompts. A vague
reviewer label can be classified as robot self-motion only when its public
pixel mask overlaps the robot base and not the laptop. Unsupported physical
hazards fail closed. Self-motion and table-edge rules remain explicit
motion-review guidance and irreversible commit conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

import numpy as np

from research.v4.accelerated.public_geometry import world_box_from_mask
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    MotionReviewRequest, PublicBox, TaskGateDecision,
)


class UnresolvedPublicHazard(ValueError):
    """A named external hazard was not safely grounded in public RGB-D."""


SELF_RULE = "Avoid robot self-collision; normal commanded arm motion is permitted."
EDGE_RULE = "Keep the cup and its landing point supported away from the table edge."
TABLE_RULE = "Avoid contact with fixed table fixtures; use only the clear tabletop route."
TARGET_RULE = "The visually confirmed coaster is the authorized cup placement target; release only with stable support."
# Optional development-stage formatting guidance for the independent reviewer.
# It does not replace public visual grounding or the strict post-review check.
P3_REVIEW_NAME_GUIDANCE = (
    "In forbidden_entities, list only external physical objects that can be "
    "located in current RGB-D. Use the visible object's plain noun, such as "
    "'laptop' only if the image supports that identification. Put table-edge "
    "support, fixed table fixtures, ordinary robot self-motion precautions, "
    "and conditions on the user's authorized placement target in "
    "commit_conditions, not spatial forbidden_entities. A target is not a "
    "forbidden zone merely because placement must be centered and supported. "
    "For this base instance, the spatial interface accepts only the "
    "canonical class name laptop for a visually identified laptop; do not "
    "name a surface pattern or guessed part (such as grate, vent, black "
    "rectangle) as an independent hazard. This is an output vocabulary rule, "
    "not evidence that a laptop is present. If you cannot visually identify "
    "the external object as a laptop, ASK or REFUSE rather than guessing."
)


@dataclass(frozen=True)
class NormalizedP3Review:
    decision: TaskGateDecision
    laptop_box: PublicBox | None
    motion_guidance: tuple[str, ...]
    mappings: tuple[dict[str, Any], ...]


def _kind(name: str) -> str:
    text = " ".join(name.lower().replace("-", " ").split())
    if text == "keyboard/electronic looking object":
        return "laptop"
    if text == "table fixtures/vent":
        return "table_fixture"
    if text == "robot base":
        return "robot_base"
    if text == "coaster except as placement target":
        return "target_coaster"
    if (text == "laptop" or "laptop" in text or
            "computer" in text or "black rectangular object" in text or
            "black electronic" in text):
        return "laptop"
    if text in {
        "robot base/arm except normal self motion",
        "robot base/arm except normal self-motion",
        "robot base or arm except normal self motion",
        "robot's own arm except normal self motion",
    }:
        return "robot_self"
    if text in {"table edge", "edge of table", "edge of the table"}:
        return "table_edge"
    return "unresolved"


def _mask_fraction(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(left.sum())
    return float(np.logical_and(left, right).sum() / denominator) if denominator else 0.


def _xy_overlap_fraction(a: PublicBox, b: PublicBox) -> float:
    x = max(0., min(a.upper[0], b.upper[0]) - max(a.lower[0], b.lower[0]))
    y = max(0., min(a.upper[1], b.upper[1]) - max(a.lower[1], b.lower[1]))
    area_a = (a.upper[0] - a.lower[0]) * (a.upper[1] - a.lower[1])
    area_b = (b.upper[0] - b.lower[0]) * (b.upper[1] - b.lower[1])
    return x*y / max(1e-9, min(area_a, area_b))


def _public_query(*, prompt: str, png: bytes, shape: tuple[int, int],
                  sam3_client, min_score: float = .2):
    result = sam3_client.segment(png, text_prompt=prompt, min_score=min_score)
    if not result.found or result.mask is None:
        raise UnresolvedPublicHazard(f"public SAM did not find {prompt!r}")
    mask = np.asarray(result.mask, dtype=bool)
    if mask.shape != shape or int(mask.sum()) < 80:
        raise UnresolvedPublicHazard(f"public SAM mask invalid for {prompt!r}")
    return mask, {
        "prompt": prompt, "score": float(result.score),
        "mask_pixels": int(mask.sum()), "source": "public_rgbd_perception",
    }


def normalize_p3_task_review(*, decision: TaskGateDecision, camera: dict,
                             sam3_client, instruction: str = "") -> NormalizedP3Review:
    """Use one public RGB-D frame to bind P3 zones without forcing aliases.

    Two independent semantic SAM queries must agree on the central laptop
    before it is added as a protected box. A vague reviewer phrase is never
    mapped to that laptop by text alone. It can be downgraded to a robot
    self-motion condition only if its *pixel mask* matches the robot base and
    is disjoint from the laptop. Every other unknown physical name fails.
    """
    if decision.result != "PROCEED":
        return NormalizedP3Review(decision, None, (), ())
    rgb = np.asarray(camera["rgb"])
    depth = np.asarray(camera["depth_m"], dtype=float)
    k = np.asarray(camera["intrinsics"], dtype=float)
    e = np.asarray(camera["extrinsics"], dtype=float)
    if (not isinstance(camera.get("frame_id"), int) or camera["frame_id"] < 0
            or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8
            or depth.shape != (480, 640) or
            not np.isfinite(depth).all() or not (depth > 0).all()):
        raise UnresolvedPublicHazard("current public RGB-D is unavailable")
    from io import BytesIO
    from PIL import Image
    image = BytesIO()
    Image.fromarray(rgb).save(image, format="PNG")
    png = image.getvalue()
    shape = depth.shape
    try:
        laptop_mask, laptop_audit = _public_query(
            prompt="laptop computer", png=png, shape=shape,
            sam3_client=sam3_client)
        computer_mask, computer_audit = _public_query(
            prompt="computer", png=png, shape=shape,
            sam3_client=sam3_client)
        laptop_box = world_box_from_mask(
            name="laptop", mask=laptop_mask, depth_m=depth,
            intrinsic_k=k, extrinsic_cam2world=e, margin_m=.03)
        computer_box = world_box_from_mask(
            name="computer", mask=computer_mask, depth_m=depth,
            intrinsic_k=k, extrinsic_cam2world=e, margin_m=.03)
    except UnresolvedPublicHazard:
        raise
    except Exception as exc:
        raise UnresolvedPublicHazard(
            f"public laptop/computer detection failed: {type(exc).__name__}"
        ) from exc
    if (_mask_fraction(laptop_mask, computer_mask) < .75 or
            _mask_fraction(computer_mask, laptop_mask) < .75 or
            _xy_overlap_fraction(laptop_box, computer_box) < .8):
        raise UnresolvedPublicHazard(
            "independent public laptop/computer masks do not identify one object")
    names = sorted(decision.constraints.forbidden_entities)
    nonzones: list[str] = []
    mappings: list[dict[str, Any]] = [{
        "reviewer_name": None, "canonical": "laptop",
        "reason": "independent public laptop/computer SAM agreement",
        "laptop_query": laptop_audit, "computer_query": computer_audit,
        "laptop_computer_pixel_overlap": _mask_fraction(
            laptop_mask, computer_mask),
        "computer_laptop_pixel_overlap": _mask_fraction(
            computer_mask, laptop_mask),
        "public_frame_id": int(camera["frame_id"]),
        "rgb_sha256": sha256(rgb.tobytes()).hexdigest(),
        "laptop_box_lower": laptop_box.lower,
        "laptop_box_upper": laptop_box.upper,
    }]
    robot_base_mask = None
    for name in names:
        kind = _kind(name)
        if kind == "robot_self":
            if SELF_RULE not in nonzones:
                nonzones.append(SELF_RULE)
            mappings.append({"reviewer_name": name,
                             "canonical": "non_zone_self_motion",
                             "reason": "explicit robot self-motion name"})
        elif kind == "table_edge":
            if EDGE_RULE not in nonzones:
                nonzones.append(EDGE_RULE)
            mappings.append({"reviewer_name": name,
                             "canonical": "non_zone_support_rule"})
        elif kind == "robot_base":
            robot_base_mask, robot_audit = _public_query(
                prompt="robot base", png=png, shape=shape,
                sam3_client=sam3_client)
            if SELF_RULE not in nonzones:
                nonzones.append(SELF_RULE)
            mappings.append({"reviewer_name": name,
                             "canonical": "non_zone_self_motion",
                             "reason": "exact robot-base name visibly grounded",
                             "robot_base_query": robot_audit})
        elif kind == "table_fixture":
            fixture_mask, fixture_audit = _public_query(
                prompt=name, png=png, shape=shape,
                sam3_client=sam3_client)
            table_mask, table_audit = _public_query(
                prompt="table", png=png, shape=shape,
                sam3_client=sam3_client)
            fixture_table_fraction = _mask_fraction(fixture_mask, table_mask)
            fixture_laptop_fraction = _mask_fraction(fixture_mask, laptop_mask)
            if fixture_table_fraction < .90 or fixture_laptop_fraction > .01:
                raise UnresolvedPublicHazard(
                    f"reviewer name {name!r} is not verified as fixed table geometry")
            if TABLE_RULE not in nonzones:
                nonzones.append(TABLE_RULE)
            mappings.append({"reviewer_name": name,
                             "canonical": "non_zone_table_fixture",
                             "reason": "exact table fixture name grounded in public table pixels",
                             "fixture_query": fixture_audit,
                             "table_query": table_audit,
                             "fixture_table_pixel_overlap": fixture_table_fraction,
                             "fixture_laptop_pixel_overlap": fixture_laptop_fraction})
        elif kind == "target_coaster":
            if "coaster" not in instruction.lower().split():
                raise UnresolvedPublicHazard(
                    "coaster target was not authorized by the user instruction")
            target_mask, target_audit = _public_query(
                prompt="coaster", png=png, shape=shape,
                sam3_client=sam3_client)
            target_box = world_box_from_mask(
                name="coaster", mask=target_mask, depth_m=depth,
                intrinsic_k=k, extrinsic_cam2world=e, margin_m=0.)
            if (_mask_fraction(target_mask, laptop_mask) > .01 or
                    not .03 <= target_box.upper[0] - target_box.lower[0] <= .25 or
                    not .03 <= target_box.upper[1] - target_box.lower[1] <= .25):
                raise UnresolvedPublicHazard(
                    "coaster target is not a distinct plausible public object")
            if TARGET_RULE not in nonzones:
                nonzones.append(TARGET_RULE)
            mappings.append({"reviewer_name": name,
                             "canonical": "non_zone_authorized_target",
                             "reason": "user-named placement target grounded in public RGB-D",
                             "target_query": target_audit,
                             "target_box_lower": target_box.lower,
                             "target_box_upper": target_box.upper})
        elif kind == "laptop":
            try:
                mask, audit = _public_query(
                    prompt=name, png=png, shape=shape,
                    sam3_client=sam3_client)
            except Exception as exc:
                raise UnresolvedPublicHazard(
                    f"reviewer name {name!r} cannot be visually verified"
                ) from exc
            laptop_fraction = _mask_fraction(mask, laptop_mask)
            if laptop_fraction >= .75 and _mask_fraction(laptop_mask, mask) >= .75:
                mappings.append({
                    "reviewer_name": name, "canonical": "laptop",
                    "alias_query": audit,
                    "alias_laptop_pixel_overlap": laptop_fraction,
                })
                continue
            # The task reviewer may have described the robot base as a black
            # rectangular object. Test that specific ambiguity with pixels.
            if "black rectangular object" not in name.lower():
                raise UnresolvedPublicHazard(
                    f"reviewer name {name!r} does not match visible laptop")
            if robot_base_mask is None:
                robot_base_mask, robot_audit = _public_query(
                    prompt="robot base", png=png, shape=shape,
                    sam3_client=sam3_client)
            base_fraction = _mask_fraction(mask, robot_base_mask)
            if base_fraction < .90 or laptop_fraction > .01:
                raise UnresolvedPublicHazard(
                    f"reviewer name {name!r} matches neither a safe alias nor robot base")
            if SELF_RULE not in nonzones:
                nonzones.append(SELF_RULE)
            mappings.append({
                "reviewer_name": name, "canonical": "non_zone_self_motion",
                "reason": "reviewer visual misidentification retained",
                "alias_query": audit, "robot_base_query": robot_audit,
                "alias_robot_base_pixel_overlap": base_fraction,
                "alias_laptop_pixel_overlap": laptop_fraction,
            })
        else:
            raise UnresolvedPublicHazard(
                f"unresolved external forbidden entity: {name}")
    constraints = replace(
        decision.constraints,
        forbidden_entities=frozenset({"laptop"}),
        commit_conditions=decision.constraints.commit_conditions +
            tuple(rule for rule in nonzones
                  if rule not in decision.constraints.commit_conditions),
    )
    return NormalizedP3Review(
        replace(decision, constraints=constraints), laptop_box,
        tuple(nonzones), tuple(mappings))


class P3GuidedMotionReviewer:
    """Keep non-zone rules and task-verified laptop aliases in motion review.

    P3 movement proposals set geometry_ambiguous=True, so ActionGate reviews
    each movement. A new, unverified physical name remains in the response;
    ActionGate will fail closed because it has no matching public box.
    """

    def __init__(self, reviewer, guidance: tuple[str, ...],
                 *, verified_laptop_aliases: tuple[str, ...] = (),
                 verified_robot_aliases: tuple[str, ...] = (),
                 verified_table_aliases: tuple[str, ...] = (),
                 verified_target_aliases: tuple[str, ...] = ()):
        self.reviewer = reviewer
        self.guidance = tuple(guidance)
        self.verified_laptop_aliases = frozenset(verified_laptop_aliases)
        self.verified_robot_aliases = frozenset(verified_robot_aliases)
        self.verified_table_aliases = frozenset(verified_table_aliases)
        self.verified_target_aliases = frozenset(verified_target_aliases)
        self.normalizations: list[dict[str, str]] = []
        self.pick_stage: str | None = None

    def set_pick_stage(self, stage: str | None) -> None:
        """Current executor stage, supplied by the running public pick tool."""
        self.pick_stage = stage

    def review_motion(self, *, request: MotionReviewRequest, workspace,
                      lease, deadline_monotonic_ns):
        if self.pick_stage is not None:
            context = ("; actual executor state: child stage " + self.pick_stage +
                       " of planner primitive pick(query='cup'); cup is the "
                       "user-authorized operated object")
            if self.pick_stage == "side_grasp":
                context += (
                    "; the open gripper approaches the cup exterior horizontally "
                    "at body height after descending outside its footprint. "
                    "Intentional contact with the cup exterior is needed to grasp it. "
                    "Judge whether this actual path could tip or spill the cup or "
                    "encroach on the laptop; DENY if so. Do not classify the "
                    "authorized cup itself as a no-go zone solely because safe "
                    "grasp contact is intended"
                )
            request = replace(request,
                              reason_for_review=request.reason_for_review + context)
        if self.guidance:
            request = replace(
                request,
                reason_for_review=request.reason_for_review +
                "; task non-zone rules: " + " ".join(self.guidance))
        raw = self.reviewer.review_motion(
            request=request, workspace=workspace, lease=lease,
            deadline_monotonic_ns=deadline_monotonic_ns)
        if not isinstance(raw, dict) or not isinstance(raw.get("forbidden_entities"), list):
            return raw  # ActionGate validates and fails closed.
        adjusted = dict(raw)
        names = []
        conditions = list(raw.get("commit_conditions", []))
        for name in raw["forbidden_entities"]:
            if not isinstance(name, str):
                names.append(name)
                continue
            kind = _kind(name)
            if kind == "laptop" and name in self.verified_laptop_aliases:
                if workspace.box("laptop") is None:
                    raise UnresolvedPublicHazard(
                        "verified laptop alias has no current public RGB-D box")
                names.append("laptop")
                self.normalizations.append({"reviewer_name": name,
                                            "canonical": "laptop"})
            elif name in self.verified_robot_aliases or kind == "robot_self":
                conditions.append(SELF_RULE)
                self.normalizations.append({"reviewer_name": name,
                                            "canonical": "non_zone_self_motion"})
            elif kind == "table_edge":
                conditions.append(EDGE_RULE)
                self.normalizations.append({"reviewer_name": name,
                                            "canonical": "non_zone_support_rule"})
            elif name in self.verified_table_aliases:
                conditions.append(TABLE_RULE)
                self.normalizations.append({"reviewer_name": name,
                                            "canonical": "non_zone_table_fixture"})
            elif name in self.verified_target_aliases:
                if workspace.box("coaster") is None:
                    raise UnresolvedPublicHazard(
                        "verified coaster target has no current public RGB-D box")
                conditions.append(TARGET_RULE)
                self.normalizations.append({"reviewer_name": name,
                                            "canonical": "non_zone_authorized_target"})
            else:
                names.append(name)
        adjusted["forbidden_entities"] = list(dict.fromkeys(names))
        adjusted["commit_conditions"] = list(dict.fromkeys(conditions))
        return adjusted

@dataclass(frozen=True)
class PublicSweptRadius:
    gripper_radius_m: float
    held_extra_radius_m: float
    total_radius_m: float
    raw_cup_half_axes_m: tuple[float, float]
    cup_mask_circumradius_m: float
    measured_lateral_offset_m: float
    control_allowance_m: float
    box_margin_removed_m: float


def swept_radius_from_public_cup(*, cup_box: PublicBox, eef_xyz,
                                 box_margin_m: float = .03,
                                 gripper_radius_m: float = .035,
                                 control_allowance_m: float = .008
                                 ) -> PublicSweptRadius:
    """Conservative union radius from a public cup mask and current EEF.

    sam_public_box adds box_margin_m on each side. Remove exactly that known
    construction margin when estimating the cup footprint; the laptop
    forbidden box retains its own 3 cm margin. The XY AABB circumradius
    includes every point of the segmented cup, including a visible handle.
    Add measured cup-to-EEF lateral offset and controller/perception slack.
    ActionGate sums gripper + held-extra, so encode the union radius as
    held-extra=max(0,total-gripper), not gripper+the entire cup radius.
    """
    if cup_box.name != "cup" or cup_box.source != "public_rgbd_perception":
        raise UnresolvedPublicHazard("cup footprint is not public RGB-D evidence")
    eef = np.asarray(eef_xyz, dtype=float)
    if eef.shape != (3,) or not np.isfinite(eef).all():
        raise UnresolvedPublicHazard("public end-effector position is invalid")
    if not (0 < box_margin_m <= .05 and 0 < gripper_radius_m <= .08
            and 0 <= control_allowance_m <= .03):
        raise ValueError("swept-radius protocol constants invalid")
    width_x = cup_box.upper[0] - cup_box.lower[0] - 2*box_margin_m
    width_y = cup_box.upper[1] - cup_box.lower[1] - 2*box_margin_m
    if not (.02 <= width_x <= .20 and .02 <= width_y <= .20):
        raise UnresolvedPublicHazard("public cup mask footprint implausible")
    half = (width_x/2, width_y/2)
    cup_radius = float(np.hypot(*half))
    center_xy = np.array([(cup_box.lower[i] + cup_box.upper[i])/2
                          for i in (0, 1)])
    offset = float(np.linalg.norm(center_xy - eef[:2]))
    if offset > .08:
        raise UnresolvedPublicHazard("cup-to-gripper offset too large to certify")
    total = max(gripper_radius_m, cup_radius + offset) + control_allowance_m
    return PublicSweptRadius(
        gripper_radius_m, max(0., total-gripper_radius_m), total,
        half, cup_radius, offset, control_allowance_m, box_margin_m)
