"""Own R304 constructor with only the reviewed two fixed camera poses changed."""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

from research.v4.accelerated.tabletop import _look_at_quat
from research.v4.candidates.r304_table_edge_presentation_20260928 import scene as own
from research.v4.candidates.r3_06_front_camera_review_20260928 import scene as rig

ROOT = Path(__file__).resolve().parents[4]
REVIEW = ROOT / 'research/v4/results/r3_06_front_camera_hd_root_review_20260928.json'
ORIGINAL_ENV = own.TableEdgePlinthCartEnv
VERSION = 'R304-lift30mm-table-edge-own-front-camera-v1-20260928'


def apply_fixed_front(worldbody):
    """The exact approved R306 pose; original fovy and wrist are retained."""
    for name, pos in [('agentview', rig.PUBLIC_POS), ('research_spectator', rig.DISPLAY_POS)]:
        camera = worldbody.find(".//camera[@name='" + name + "']")
        if camera is None:
            raise ValueError('Registered own camera missing ' + name)
        camera.set('pos', ' '.join(map(str, pos)))
        camera.set('quat', ' '.join(map(str, _look_at_quat(pos, rig.TARGET))))


class FrontTableEdgeEnv(ORIGINAL_ENV):
    def _load_model(self):
        super()._load_model()
        apply_fixed_front(self.model.worldbody)


@contextmanager
def front_scope():
    from hashlib import sha256
    review = json.loads(REVIEW.read_text())
    key = str(Path(rig.__file__).relative_to(ROOT))
    if not review['accepted'] or review['source_sha256'].get(key) != sha256(Path(rig.__file__).read_bytes()).hexdigest():
        raise RuntimeError('Approved fixed-camera source changed')
    if own.TableEdgePlinthCartEnv is not ORIGINAL_ENV:
        raise RuntimeError('Own R304 source/plinth constructor seam changed')
    with patch.object(own, 'TableEdgePlinthCartEnv', FrontTableEdgeEnv):
        yield


class FrontFoilScene(own.TableEdgeFoilScene):
    public_camera_revision = VERSION

    def __init__(self, *args, **kwargs):
        with front_scope():
            super().__init__(*args, **kwargs)
