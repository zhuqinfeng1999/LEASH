"""Regression for #234: a blocked crossing must suggest the qualified rear lane."""
from types import SimpleNamespace

from research.v4.accelerated.p1_gated_toolkit import P1GatedToolkit
from research.v4.candidates.leash_commit_gate_20260924.commit_gate import (
    CameraFrame, PublicEvidence, PublicRobotState,
)
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    PublicBox, PublicWorkspace, TaskConstraints, TaskGateDecision,
)


class _Base:
    def __init__(self):
        self.calls = []

    def execute_tool(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(result={"ok": True})


class _Adapter:
    last_outcome = None

    def set_commit_conditions(self, _):
        pass


class _Reviewer:
    def review_motion(self, **_):
        return {"decision": "APPROVE", "reason": "clear",
                "forbidden_entities": [], "commit_conditions": []}


def test_real_public_seed1_block_gives_moderate_rear_route_with_no_motion():
    frame = CameraFrame("agentview", b"rgb", b"depth", 640, 480, {}, {}, 1)
    wrist = CameraFrame("robot0_eye_in_hand", b"rgb", b"depth", 640, 480, {}, {}, 1)
    evidence = PublicEvidence(frame, wrist, PublicRobotState(False, "held"), 1)
    parts = (
        PublicBox("left_hand", (.102, -.466, .870), (.326, -.284, 1.001)),
        PublicBox("right_hand", (.000, -.027, .870), (.229, .173, 1.001)),
        PublicBox("left_forearm", (.267, -.431, .921), (.528, -.291, 1.115)),
        PublicBox("forearm", (.165, .033, .870), (.442, .184, 1.080)),
    )
    workspace = PublicWorkspace(evidence, (-.06, -.13, 1.10), parts)
    task = TaskGateDecision("PROCEED", "avoid person",
                            TaskConstraints(allowed_operated_entities=frozenset({"knife"}),
                                            forbidden_entities=frozenset(p.name for p in parts)),
                            False)
    base = _Base()
    toolkit = P1GatedToolkit(
        base, _Adapter(), round_id="P1-01-seed1", task_decision=task,
        reviewer=_Reviewer(), workspace=lambda: workspace,
        locate_vla_target=lambda *_: None,
        public_context=lambda: {"robot_state": {
            "grasp_latched": True, "eef_pos": (-.06, -.13, 1.10),
            "eef_quat": (1., 0., 0., 0.),
        }},
    )
    denied = toolkit.execute_tool("move_pose", {
        "xyz": [.22, .27, 1.10], "gripper": 1,
        "target_pitch": None, "target_yaw": None,
    })
    assert denied.result["code"] == "action_denied"
    assert "x=-0.250 m" in denied.result["constraint"]
    assert "Avoid x<-.30 m" in denied.result["constraint"]
    assert base.calls == []
