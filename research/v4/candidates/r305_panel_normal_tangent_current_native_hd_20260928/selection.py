"""Current own1230 factory; only existing fixed front cameras are added."""
import hashlib
import json
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

from research.v4.accelerated.tabletop import _look_at_quat
from research.v4.candidates.r3_06_front_camera_review_20260928.scene import PUBLIC_POS, DISPLAY_POS, TARGET
from research.v4.candidates.r305_panel_normal_tangent_actual_20260928 import entry as current
from research.v4.candidates.r305_robot_facing_lug_y80_copy_20260928 import scene as own

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
APPROVED = current.H / "source_freeze.json"
CAMERAS = (("agentview", PUBLIC_POS), ("research_spectator", DISPLAY_POS))
LOCAL = ("selection.py", "native_hd.py", "contracts.py", "freeze.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(pins):
    parent = json.loads(APPROVED.read_text())
    required = {str((H / name).relative_to(ROOT)) for name in LOCAL}
    required.add(str(APPROVED.relative_to(ROOT)))
    if not required.issubset(pins) or any(pins.get(name) != digest for name, digest in parent.items()):
        raise RuntimeError("Exact whole own1230/current native sources required")
    for name, digest in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError("Current own source drift: " + name)
    return pins


def admission():
    return validate(json.loads((H / "source_freeze.json").read_text())["source_sha256"])


ORIGINAL_LOADER = own.ORIGINAL_LOAD


def front_loader(env):
    ORIGINAL_LOADER(env)
    for name, position in CAMERAS:
        nodes = env.model.worldbody.findall(".//camera[@name='" + name + "']")
        if len(nodes) != 1:
            raise RuntimeError("Exact existing fixed front camera missing: " + name)
        nodes[0].set("pos", " ".join(map(str, position)))
        nodes[0].set("quat", " ".join(map(str, _look_at_quat(position, TARGET))))


_INIT = FunctionType(own.RobotFacingLugY80Scene.__init__.__code__,
                     dict(own.RobotFacingLugY80Scene.__init__.__globals__, ORIGINAL_LOAD=front_loader),
                     closure=own.RobotFacingLugY80Scene.__init__.__closure__)


def native_selected(output, *, render_device=0):
    admission()
    original, source = current.selected()
    factory = original.__globals__["R3TabletopFacade"]
    if factory is not own.RobotFacingLugY80Scene:
        raise RuntimeError("Own current factory identity changed")

    class CurrentFrontNormalTangentForkBowlScene(factory):
        def __init__(self, *args, **kwargs):
            admission()
            kwargs["camera_observations"] = True
            kwargs["render_gpu_device_id"] = render_device
            if own.parent.ENV._load_model is not ORIGINAL_LOADER:
                raise RuntimeError("Current own loader is already scoped")
            with patch.object(own.parent.ENV, "_load_model", front_loader):
                _INIT(self, *args, **kwargs)
            if own.parent.ENV._load_model is not ORIGINAL_LOADER:
                raise RuntimeError("Native-only camera loader scope did not restore")

    return FunctionType(original.__code__, dict(original.__globals__,
                        R3TabletopFacade=CurrentFrontNormalTangentForkBowlScene),
                        closure=original.__closure__), source
