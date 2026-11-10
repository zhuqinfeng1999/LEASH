"""Ground S1 gate-2 entities from either declared public RGB-D camera.

The original S1 workspace tried agentview only. At the parked pose the left
forearm can be cropped there while it is visible in the wrist camera. This
fallback only changes perception; a name missing in both views remains
absent, which makes the action gate fail closed.
"""
from __future__ import annotations

from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicWorkspace
from research.v4.candidates.s1_multiround_driver_20260925.public_grounding import (
    PROMPTS, S1PublicWorkspace,
)


class S1DualViewWorkspace(S1PublicWorkspace):
    def __call__(self) -> PublicWorkspace:
        packet = self.scene.capture_agent()
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
            failures = []
            for camera_name in ("agentview", "robot0_eye_in_hand"):
                try:
                    box, details = sam_public_box(
                        name=name, prompt=prompt,
                        camera=packet["cameras"][camera_name],
                        sam3_client=self.sam3_client)
                    boxes.append(box)
                    self.log.append({"frame_id": packet["frame_id"],
                                     "camera": camera_name, **details})
                    break
                except Exception as exc:
                    failures.append(f"{camera_name}: {type(exc).__name__}: {exc}")
            else:
                self.log.append({"frame_id": packet["frame_id"],
                                 "name": name, "error": "; ".join(failures)})
        eef = tuple(float(x) for x in packet["robot_state"]["eef_pos"])
        evidence = self.adapter.capture_commit_evidence()
        return PublicWorkspace(evidence, eef, tuple(boxes))
