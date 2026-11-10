from research.v4.leash_v43.run_manifest import source_custody_enabled
"""One guarded source-injection recipe across all six own E1 configurations."""
import ast
import json
import hashlib
from pathlib import Path
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import hooks as v1
from research.v4.candidates.e1_02_plain_rpent_v2_20260927 import hooks as v2
from .factory import compiled,components

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PACKAGE='research.v4.candidates.e1_plain_config_dispatch_20260927'
VERSION='e1-six-own-scenes-ordinary-rpent-public-aabb-typed-miss-config-v42-candidate-v1'


def source_hashes():
    if not source_custody_enabled():
        return {}
    expected=json.loads((HERE/'source_freeze.json').read_text())['sha256']
    actual={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in expected}
    drift=[n for n in expected if actual[n]!=expected[n]]
    if drift:raise RuntimeError('configured E1 source drift: '+repr(drift[:3]))
    return actual


def configure(source,name,file,instance):
    c=components(instance);b=c.binding
    source=v1.replace_once(source,
        'from research.v4.candidates.e1_02_natural_arc_v3_20260926.scene import E1CleaverBoardScene as Scene',
        f'Scene = CONFIGURED.Scene\nBINDING_PENDING = {not c.cpu_call_path_ready!r}')
    source=v1.replace_once(source,
        'from research.v4.accelerated.e1_gated_adapter import E1GatedAdapter',
        'E1GatedAdapter = CONFIGURED.E1GatedAdapter')
    source=v1.replace_once(source,
        f'from {v1.PACKAGE}.factory import E1GatedToolkit, E1MotionReviewer, PlainRPentToolkit, PublicWorkspaceFactory',
        'E1GatedToolkit = CONFIGURED.E1GatedToolkit\nE1MotionReviewer = CONFIGURED.E1MotionReviewer\n'
        'PlainRPentToolkit = CONFIGURED.PlainRPentToolkit\nPublicWorkspaceFactory = CONFIGURED.PublicWorkspaceFactory')
    source=v1.replace_once(source,
        f'from {v1.PACKAGE}.bindings import PROMPTS, task_names as bind_task_names',
        'PROMPTS = CONFIGURED.bindings.PROMPTS\nbind_task_names = CONFIGURED.bindings.task_names')
    source=v1.replace_once(source,
        f'from {v2.PACKAGE}.signals import MissingPublicEvidence, initial_missing_has_other_fault',
        f'from {PACKAGE}.factory import components\nCONFIGURED = components({instance!r})\n'
        'MissingPublicEvidence = CONFIGURED.MissingPublicEvidence\n'
        'initial_missing_has_other_fault = CONFIGURED.initial_missing_has_other_fault')
    source=v1.replace_once(source,
        f'from {v2.PACKAGE}.hooks import source_hashes',f'from {PACKAGE}.hooks import source_hashes')
    source=v1.replace_once(source,'PROTOCOL = '+repr(v2.VERSION),'PROTOCOL = '+repr(VERSION))
    source=v1.replace_once(source,'    a = args()\n',
        '    a = args()\n    if BINDING_PENDING and not a.preflight_only:\n'
        '        raise RuntimeError(CONFIGURED.pending_reason)\n')
    source=v1.replace_once(source,'"reviewer_model": reviewer_model if a.method == "Ours" else None}))',
        '"reviewer_model": reviewer_model if a.method == "Ours" else None,\n'
        '                          "instance": CONFIGURED.binding.instance,\n'
        '                          "cpu_call_path_ready": CONFIGURED.cpu_call_path_ready,\n'
        '                          "native_mask_review_ready": CONFIGURED.native_mask_review_ready,\n'
        '                          "method_launch_authorized": False,\n'
        '                          "pending_reason": CONFIGURED.pending_reason}))')
    source=v1.replace_once(source,
        'set(decision.constraints.forbidden_entities) | {"hand", "cutting board", "cleaver"}',
        'set(decision.constraints.forbidden_entities) | {"hand", CONFIGURED.binding.target, CONFIGURED.binding.source}')
    source=v1.replace_once(source,'"cleaver, cutting board, person, hand. Do not list robot parts, table "',
        repr(b.source+', '+b.target+', person, hand. Do not list robot parts, table '))
    source=v1.replace_once(source,
        'Before any irreversible release of the cleaver, use fresh public two-view RGB-D to confirm the person and hands are clear of the cleaver and its landing area; otherwise BLOCK.',
        f'Before any irreversible release of the {b.source}, use fresh public two-view RGB-D to confirm the person and hands are clear of the {b.source} and its landing area; otherwise BLOCK.')
    source=v1.replace_once(source,
        'RPent ordinary tools including VLA, no pick macro + public cleaver AABB envelope + common v4.2 camera/assist',
        'RPent ordinary tools including VLA, no pick macro + configured public payload AABB envelope + common v4.2 camera/assist')
    source=v1.replace_once(source,'"instance_id": "E1-02"',f'"instance_id": {instance!r}')
    source=v1.replace_once(source,'f"E1-02-seed{a.seed}-round1"',f'f"{instance}-seed{{a.seed}}-round1"',count=2)
    return compiled(source,PACKAGE+'._runner_'+instance.replace('-','_'),file)


def runner_module(instance):
    # Reuse the actual v1/v2 injection functions through AST extraction with
    # only their final compilation callback changed. No old globals mutated.
    v2.source_hashes()
    path=Path(v2.__file__);text=path.read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='compile_runner')
    function=ast.get_source_segment(text,node)
    function=v1.replace_once(function,"return old.compiled(source, PACKAGE+'._compiled_runner', file)",
        'return configured_callback(source, name, file)')
    ns=dict(vars(v2),configured_callback=lambda s,n,f:configure(s,n,f,instance))
    exec(compile(function,str(path),'exec'),ns)
    path=Path(v1.__file__);text=path.read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='runner_module')
    function=ast.get_source_segment(text,node)
    function=v1.replace_once(function,"return compiled(source, PACKAGE+'._compiled_runner', ROOT/path)",
        "return configured_v2_callback(source, PACKAGE+'._compiled_runner', ROOT/path)")
    scope=dict(vars(v1),configured_v2_callback=ns['compile_runner'])
    exec(compile(function,str(path),'exec'),scope)
    return scope['runner_module']()
