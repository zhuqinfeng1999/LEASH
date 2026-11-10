"""Only two existing fixed front cameras, on the current five-box ledge."""
from types import FunctionType

import mujoco
import numpy as np

from research.v4.accelerated.tabletop import _look_at_quat
from research.v4.candidates.r3_06_front_camera_review_20260928.scene import PUBLIC_POS, DISPLAY_POS, TARGET
from research.v4.candidates.r302_flat_ledge_source_20260928.scene import FlatLedgeTall350Env
from research.v4.candidates.r302_flat_ledge_guard_metadata_v2_20260928 import guard as physical_guard

CAMERAS = (("agentview", PUBLIC_POS), ("research_spectator", DISPLAY_POS))
CAMERA_FIELDS = ("cam_pos", "cam_quat", "cam_pos0", "cam_poscom0", "cam_mat0")


class FrontFlatLedgeTall350Env(FlatLedgeTall350Env):
    def _load_model(self):
        super()._load_model()
        for name, position in CAMERAS:
            nodes = self.model.worldbody.findall(".//camera[@name='" + name + "']")
            if len(nodes) != 1:
                raise RuntimeError("Exact existing fixed camera missing: " + name)
            nodes[0].set("pos", " ".join(map(str, position)))
            nodes[0].set("quat", " ".join(map(str, _look_at_quat(position, TARGET))))


def camera_reference(expected, receipt, paths):
    """Declared fixed-world pose and compiler COM arithmetic; never copy actual."""
    changes = []
    com = np.asarray(receipt["qpos0_world_COM"], dtype=np.float64)
    for name, position in CAMERAS:
        index = expected.camera(name).id
        if expected.cam_bodyid[index] != 0:
            raise RuntimeError("Existing front camera is no longer world-mounted: " + name)
        old = {field: getattr(expected, field)[index].copy().tolist() for field in CAMERA_FIELDS}
        quat = np.asarray([float(str(v)) for v in _look_at_quat(position, TARGET)])
        mujoco.mju_normalize4(quat)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quat)
        expected.cam_pos[index] = position
        expected.cam_quat[index] = quat
        expected.cam_pos0[index] = position
        expected.cam_poscom0[index] = np.asarray(position) - com
        expected.cam_mat0[index] = matrix.reshape(expected.cam_mat0[index].shape)
        changes.append(dict(name=name, id=index, before=old, position=list(position),
                            quaternion=quat.tolist(), target=list(TARGET)))
    receipt = dict(receipt, explicit_front_camera_changes=changes,
                   camera_fields=list(CAMERA_FIELDS), original_wrist_and_fovy_unchanged=True,
                   public_resolution=[640, 480], no_actual_model_values_copied=True)
    return expected, receipt, paths


def reference(seed):
    return camera_reference(*physical_guard.reference(seed))


_COMPARE = FunctionType(physical_guard.compare.__code__,
                        dict(physical_guard.compare.__globals__, reference=reference),
                        closure=physical_guard.compare.__closure__)


def compare_native(actual, seed):
    result = _COMPARE(actual, seed)
    result.update(selected_physical_guard="994-flat-ledge-metadata-v2",
                  camera_revision="existing-R306-fixed-front-camera-20260928",
                  explicit_camera_rows_only=True, all_other_model_fields_checked=True,
                  group_changes=0, no_runtime_model_or_state_writes=True)
    return result
