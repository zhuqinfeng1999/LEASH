"""Bridge-only I1 public packet normalization; accepted task Scene stays frozen."""
from collections.abc import Mapping


class I1PublicSceneFacade:
    """Expose only accepted I1 RGB-D and robot state to the existing adapter."""

    def __init__(self, scene):
        self._scene = scene

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def capture_agent(self):
        packet = self._scene.capture_agent()
        if not isinstance(packet, Mapping) or set(packet) != {"cameras", "robot_state"}:
            raise ValueError("I1 public packet has extra or missing fields")
        cameras = packet["cameras"]
        if not isinstance(cameras, Mapping) or set(cameras) != {
                "agentview", "robot0_eye_in_hand"}:
            raise ValueError("I1 public packet must have exactly two RGB-D cameras")
        for camera in cameras.values():
            if not isinstance(camera, Mapping) or set(camera) != {"rgb", "depth_m"}:
                raise ValueError("I1 camera packet has extra or missing fields")
        robot = packet["robot_state"]
        if not isinstance(robot, Mapping) or set(robot) != {
                "eef_pos", "eef_quat", "gripper_qpos"}:
            raise ValueError("I1 robot state has extra or missing fields")
        return {"cameras": cameras, **robot}


def make_private_evaluator(module, task, instance_id):
    """Bind I1 truth to the selected instance; preserve R evaluator signatures."""
    return module.Evaluator(instance_id) if task == "i1" else module.Evaluator()
