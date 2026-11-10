"""Opt-in R303 private end-check and already-reviewed fixed camera binding."""
import hashlib
import json
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from research.v4.candidates.r303_seed2_overhead_actual_20260928 import run_once as previous
from research.v4.candidates.r303_overhead_qualification10_20260928 import qualify as old_qualification
from research.v4.candidates.scene_review_r303_overhead_base90_20260928 import export as old_native
from research.v4.candidates.r3_03_own_front_carrier_20260928 import scene as own_front
from research.v4.candidates.r3_06_front_camera_review_20260928 import scene as reviewed_camera
from research.v4.candidates.r303_base_forward90_copy_20260928.scene import BaseForward90OwnThermosScene

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
OLD_SUMMARY = old_native.QUALIFICATION
CAMERA_REVIEW = ROOT / 'research/v4/results/r3_06_front_camera_hd_root_review_20260928.json'
VERSION = 'R303-base90-overhead-actual-site-endcheck-R306-fixed-front-cameras-v1'
ORIGINAL_CART_ENV = own_front.OwnCartEnv


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def actual_driver_source():
    source = previous.expert.generic_driver_source()
    old = "np.asarray(scene.raw['robot0_eef_pos'])"
    if source.count(old) != 2:
        raise ValueError('Expected exactly two cached private stage-summary reads')
    # The loop and rotation summary already use these actual site fields.
    # No goal/tolerance/action/budget/controller/online getter changes.
    return source.replace(old, 'np.asarray(d.site_xpos[center])')


def selected():
    _, original, case_source, naive, naive_source = previous.admission()
    namespace = dict(naive.__globals__, generic_driver_source=actual_driver_source)
    corrected = types.FunctionType(naive.__code__, namespace, naive.__name__,
                                   naive.__defaults__, naive.__closure__)
    return hashes(), original, case_source, corrected, naive_source


@contextmanager
def front_camera_scope():
    review = json.loads(CAMERA_REVIEW.read_text())
    key = str(Path(reviewed_camera.__file__).relative_to(ROOT))
    if not review['accepted'] or review['source_sha256'].get(key) != sha(reviewed_camera.__file__):
        raise ValueError('Actual reviewed R306 fixed camera source changed')
    if (reviewed_camera.FrontCartEnv.__mro__[1] is not ORIGINAL_CART_ENV
            or own_front.OwnCartEnv is not ORIGINAL_CART_ENV):
        raise ValueError('Own original cart constructor seam changed')
    with patch.object(own_front, 'OwnCartEnv', reviewed_camera.FrontCartEnv):
        yield


class OwnFrontCameraScene(BaseForward90OwnThermosScene):
    public_camera_revision = VERSION

    def __init__(self, seed=0, *, camera_observations=False, render_gpu_device_id=-1):
        with front_camera_scope():
            super().__init__(seed, camera_observations=camera_observations,
                             render_gpu_device_id=render_gpu_device_id,
                             root_base_forward_approved=True)


def physical_reuse_rows():
    """Exact own20 known-script results; not newly executed or camera-qualified."""
    summary = json.loads(OLD_SUMMARY.read_text())
    old_qualification.admission()
    if not summary['complete'] or (summary['k'], summary['n']) != (30, 30):
        raise ValueError('Original same-physical 30/30 evidence missing')
    _, _, source, _, naive_source = previous.admission()
    if (summary['generated_case_sha256'] != hashlib.sha256(source.encode()).hexdigest()
            or summary['overhead_naive_sha256'] != hashlib.sha256(naive_source.encode()).hexdigest()):
        raise ValueError('Original evidence code changed')
    rows = []
    for row in summary['rows']:
        if row['mode'] == 'naive':
            continue
        p = Path(row['result_path'])
        if not row['passed'] or sha(p) != row['result_sha256']:
            raise ValueError('Original correct/capability raw changed')
        rows.append(dict(row, reused_exact={'path': str(p), 'sha256': sha(p)},
                         reuse_scope='original branch and nonvisual layout only; requires own camera model pair'))
    if len(rows) != 20:
        raise ValueError('Expected only10 correct +10 capability original results')
    return rows


def hashes():
    pins = dict(old_native.hashes())
    files = [*H.glob('*.py'), H/'protocol.json', H/'cpu_contract.json', H/'actual_private_driver.py.txt',
             CAMERA_REVIEW, Path(reviewed_camera.__file__), OLD_SUMMARY,
             ROOT/'research/v4/candidates/user_rework_r4_20260927/r401_public_workspace_v6/check_cpu.py',
             old_native.H/'capture/report.json',
             ROOT/'research/v4/candidates/r303_native_cached_eef_readonly_20260928/source_freeze.json',
             ROOT/'research/v4/candidates/r303_native_cached_eef_readonly_20260928/saved_comparison.json']
    for p in files:
        if not p.is_file():
            raise ValueError('Candidate evidence missing '+str(p))
        key = str(p.relative_to(ROOT))
        digest = sha(p)
        if key in pins and pins[key] != digest:
            raise ValueError('Original/reviewed dependency conflict '+key)
        pins[key] = digest
    pins = dict(sorted(pins.items()))
    if (H/'source_freeze.json').exists() and pins != json.loads((H/'source_freeze.json').read_text()):
        raise ValueError('New isolated source admission drift')
    return pins


def camera_pair_admission():
    report = json.loads((H/'root_camera_pair/report.json').read_text())
    if (not report['passed'] or report['source_sha256'] != hashes()
            or not report['nonvisual_model_exact'] or not report['state_controller_contact_exact']
            or not report['wrist_calibration_exact']):
        raise ValueError('Actual own camera-only constructor proof missing or failed')
    return report
