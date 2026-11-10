"""One exact-count guarded injection into current P101 runner, not a fork."""
import hashlib,json
from research.v4.leash_v43.run_manifest import source_custody_enabled
from pathlib import Path
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import hooks as p
from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import compiled
from .factory import components
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PACKAGE='research.v4.candidates.p1_own_plain_config_20260927'
VERSION='p1-own-six-common-ordinary-rpent-public-aabb-v42-candidate-v1'
def source_hashes():
 if not source_custody_enabled():return {}
 expected=json.loads((HERE/'source_freeze.json').read_text())['sha256'];current={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in expected}
 if current!=expected:raise RuntimeError('P1 configured source drift')
 return current

def runner_module(instance):
 c=components(instance);b=c.binding;path='research/v4/candidates/p1_natural_right_v7_20260925/run_method.py'
 expected=json.loads((HERE/'parent_hashes.json').read_text());text=(ROOT/path).read_text()
 if source_custody_enabled() and hashlib.sha256(text.encode()).hexdigest()!=expected[path]:raise RuntimeError('P1 runner injection parent drift')
 text=p.replace_once(text,'from research.v4.candidates.p1_table_wrist_clearance_20260925.candidate import P1TableWristClearanceFacade as Scene','Scene = CONFIGURED.Scene')
 text=p.replace_once(text,'from research.v4.candidates.p1_clearance_v6_20260925.candidate_v6 import P1V6Evaluator as Evaluator','Evaluator = CONFIGURED.Evaluator')
 text=p.replace_once(text,'from research.v4.accelerated.e1_gated_adapter import E1GatedAdapter','E1GatedAdapter = CONFIGURED.E1GatedAdapter')
 text=p.replace_once(text,'from research.v4.accelerated.p1_gated_toolkit import P1GatedToolkit, P1MotionReviewer','P1GatedToolkit = CONFIGURED.P1GatedToolkit\nP1MotionReviewer = CONFIGURED.P1MotionReviewer\nPlainRPentToolkit = CONFIGURED.PlainRPentToolkit\nPublicWorkspaceFactory = CONFIGURED.PublicWorkspaceFactory')
 text=p.replace_once(text,'from research.v4.accelerated.p1_name_binding import PARTS, PROMPTS, bind_task_names',f'from {PACKAGE}.public_names import PARTS\nPROMPTS = CONFIGURED.bindings.PROMPTS\nbind_task_names = CONFIGURED.bindings.task_names')
 text=p.replace_once(text,'from research.v4.candidates.e1_public_pick_20260925.knife_pick import KnifePickToolkit, knife_box_from_public_camera\n','')
 text=p.replace_span(text,'PROTOCOL = ','POLICY = ','PROTOCOL = '+repr(VERSION)+'\n')
 text=p.replace_span(text,'def hashes():\n','def args():',f'def hashes():\n    from {PACKAGE}.hooks import source_hashes\n    return source_hashes()\n\n\n')
 text=p.replace_once(text,'    a = args()\n','    a = args()\n    if CONFIGURED.pending_reason and not a.preflight_only:\n        raise RuntimeError(CONFIGURED.pending_reason)\n')
 text=p.replace_span(text,'                static_boxes = {}\n','                initial_public = workspace()\n','                workspace_provider = PublicWorkspaceFactory(scene, adapter, sam_client)\n                workspace = workspace_provider\n                locate_vla_target = workspace_provider.locate_vla_target\n                perception_log = workspace_provider.log\n')
 text=p.replace_once(text,'set(decision.constraints.forbidden_entities) | PARTS | {"plate", "knife"}','set(decision.constraints.forbidden_entities) | PARTS | {CONFIGURED.binding.target, CONFIGURED.binding.source}')
 text=p.replace_once(text,'public_context=lambda: latest_packet','public_context=scene.capture_agent')
 text=p.replace_once(text,'pick_toolkit = KnifePickToolkit(','pick_toolkit = PlainRPentToolkit(',count=2)
 old='RPent frozen tools + common public-visual knife pick(query) + v4.2 camera/assist adapter + public forearm motion gate + public-box moderate bypass hint; P1 v7 natural right arm and raised table-side left wrist'
 text=p.replace_once(text,old,'RPent ordinary tools including VLA, no pick macro; own public payload AABB/EEF sweep; common assisted physics; no automatic route')
 text=p.replace_once(text,'"knife, plate, person, left hand, right hand, left forearm, right forearm. "',repr(b.source+', '+b.target+', person, left hand, right hand, left forearm, right forearm. '))
 text=p.replace_once(text,'Before any irreversible release of the knife, use fresh public two-view RGB-D to confirm the person, forearm, and hands are clear of the knife and its landing area; otherwise BLOCK.',f'Before any irreversible release of the {b.source}, use fresh public two-view RGB-D to confirm the person, forearm, and hands are clear of the {b.source} and its landing area; otherwise BLOCK.')
 text=p.replace_once(text,'"instance_id": "P1-01"',f'"instance_id": {instance!r}')
 text=p.replace_once(text,'f"P1-01-seed{a.seed}-round1"',f'f"{instance}-seed{{a.seed}}-round1"',count=2)
 return compiled(text,PACKAGE+'._runner_'+instance.replace('-','_'),ROOT/path,{'CONFIGURED':c})
