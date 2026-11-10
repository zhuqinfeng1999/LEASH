"""Current R201 assets at the existing ordinary T1 three-gate method seams.

The stable legacy carrier retains its real constructor/reset/controller/assist.
This candidate does not migrate physics, run models, or change old methods.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import FunctionType

from research.v4.accelerated.e1_facade import E1TabletopFacade
from research.v4.candidates.user_rework_r2_wirepaper_ember_v2_20260928.binding import R201WirePaperScene
from research.v4.candidates.r2_02_natural_method_20260926 import run_method as original
from research.v4.candidates.r2_02_natural_method_20260926 import public_boundary as public
from research.v4.candidates.r2_02_natural_method_20260926 import review_worker
from research.v4.candidates.r2_02_t1_derived_20260926.offline_evaluator import Evaluator
from research.v4.candidates.e1_unknown_tilt_typed_20260927.outlet import TypedContainerAdapter, commit_infrastructure_failure
from . import sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = 'R201-current-wire-bin-original-carrier-common-T1-three-gates-typed-axis-v1'
NOUNS = {'item': 'lit candle', 'box': 'trash can'}
ALIASES = dict(item='item', candle='item', **{'lit candle': 'item', 'burning candle': 'item',
               'trash can': 'box', 'trash bin': 'box', 'paper bin': 'box', 'bin': 'box', 'box': 'box'})


class CurrentReviewerWorker(review_worker.T1ReviewerWorker):
    """Module-level factory remains importable by the independent reviewer spawn."""
    review = FunctionType(review_worker.T1ReviewerWorker.review.__code__,
                         dict(vars(review_worker), PUBLIC_NOUNS=NOUNS), 'review',
                         closure=review_worker.T1ReviewerWorker.review.__closure__)


class CurrentR201Scene(R201WirePaperScene):
    item_name = 'lit_candle_1'
    _item_bottom_z = E1TabletopFacade._item_bottom_z
    public_grasp_latched = E1TabletopFacade.public_grasp_latched
    capture_agent = E1TabletopFacade.capture_agent

    def __init__(self, seed=0, *, camera_observations=True, render_gpu_device_id=0):
        if camera_observations is not True:
            raise ValueError('This existing legacy carrier always renders its two public cameras')
        if int(os.environ.get('MUJOCO_EGL_DEVICE_ID', '0')) != render_gpu_device_id:
            raise ValueError('Legacy carrier device argument/environment differ')
        # No extra reset, step, forward, qpos or equality modification.
        super().__init__(instance_id='R2-01', seed=seed)
        self.initial_item_bottom_z = self._item_bottom_z()


def qualification(*, require_native=False):
    sources.admit()
    acceptance = json.loads((ROOT / 'research/v4/results/r2_01_user_revision_v42_instance_acceptance_20260928.json').read_text())
    physical = json.loads((ROOT / acceptance['physical_identity']).read_text())
    if not acceptance['accepted'] or not physical['accepted']:
        raise RuntimeError('Own current R201 revision is not accepted')
    if not any(r['instance'] == 'R2-01' and r['exact'] for r in physical['rows']):
        raise RuntimeError('Current wire-bin own nonvisual constructor identity missing')
    if require_native:
        raise RuntimeError('Current R201 common packet and full three-gate method integration remain pending')
    return dict(current_scene_revision_accepted=True,
                own_prior_physical_qualification_reuse=True,
                current_asset_native_sidecar=acceptance['native_sidecar'],
                method_packet_and_three_gate_integration_pending=True,
                native_method_ready=False, launch_authorized=False)


def build_entry(*, version=VERSION):
    if version != VERSION:
        raise ValueError('Unknown current R201 method version')
    sources.admit()
    p = dict(vars(public), PUBLIC_NOUNS=NOUNS, ALIASES=ALIASES)
    bind_name = FunctionType(public.bind_name.__code__, p, 'bind_name')
    p['bind_name'] = bind_name
    bind_task = FunctionType(public.bind_task.__code__, p, 'bind_task')
    class CurrentWorkspace(public.PublicWorkspaceReader):
        __call__ = FunctionType(public.PublicWorkspaceReader.__call__.__code__, p,
                                '__call__', closure=public.PublicWorkspaceReader.__call__.__closure__)
        locate_vla_target = FunctionType(public.PublicWorkspaceReader.locate_vla_target.__code__, p,
                                        'locate_vla_target', closure=public.PublicWorkspaceReader.locate_vla_target.__closure__)
    class CurrentMotion(public.PublicMotionReviewer):
        review_motion = FunctionType(public.PublicMotionReviewer.review_motion.__code__, p,
                                     'review_motion', closure=public.PublicMotionReviewer.review_motion.__closure__)
    # This existing module-level worker reads PUBLIC_NOUNS at each review.
    # Give it an independent namespace rather than patch the parent during spawn.
    ns = dict(vars(original), Scene=CurrentR201Scene, Evaluator=Evaluator,
              INSTANCE='R2-01', INSTRUCTION=CurrentR201Scene.instruction,
              PROTOCOL=VERSION, hashes=sources.admit, qualification=qualification,
              E1GatedAdapter=TypedContainerAdapter, T1ReviewerWorker=CurrentReviewerWorker,
              PublicWorkspaceReader=CurrentWorkspace, PublicMotionReviewer=CurrentMotion,
              bind_task=bind_task, commit_infrastructure_failure=commit_infrastructure_failure)
    text = inspect.getsource(original.main)
    old = "        actual_commit_fault=any(x.get('system_failure') for x in (report.get('commit_gate') or {}).get('logs',[]))"
    if text.count(old) != 1:
        raise RuntimeError('Original actual commit-finalizer seam changed')
    text = text.replace(old,
        "        commit_rows=(report.get('commit_gate') or {}).get('logs',[])\n"
        "        report['public_capability_uncertainties']=[x for x in commit_rows if x.get('failure_kind')=='public_capability_unknown_tilt']\n"
        "        actual_commit_fault=commit_infrastructure_failure(commit_rows)", 1)
    exec(compile(text, str(HERE / 'binding.py') + '::existing-T1-main', 'exec'), ns)
    runner = ns['main']
    runner.selection_text = text
    runner.native_method_ready = False
    runner.launch_authorized = False
    return runner
