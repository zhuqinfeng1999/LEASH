from research.v4.leash_v43.run_manifest import source_custody_enabled
"""Reuse the v1 hook function itself, with one scoped compilation callback."""
import ast
import hashlib
import json
from pathlib import Path
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import hooks as old

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKAGE = 'research.v4.candidates.e1_02_plain_rpent_v2_20260927'
VERSION = 'e1-02-plain-rpent-public-aabb-typed-initial-miss-v42-candidate-v2'


def source_hashes():
    if not source_custody_enabled():
        return {}
    expected = json.loads((HERE/'source_freeze.json').read_text())['sha256']
    actual = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in expected}
    changed = [name for name in expected if expected[name] != actual[name]]
    if changed: raise RuntimeError('candidate source drift: '+repr(changed[:3]))
    return actual


def compile_runner(source, name, file):
    source = old.replace_once(source, 'from __future__ import annotations',
        'from __future__ import annotations\n'+
        f'from {PACKAGE}.signals import MissingPublicEvidence, initial_missing_has_other_fault')
    source = old.replace_once(source, 'PROTOCOL = '+repr(old.VERSION), 'PROTOCOL = '+repr(VERSION))
    source = old.replace_once(source,
        f'from {old.PACKAGE}.hooks import source_hashes', f'from {PACKAGE}.hooks import source_hashes')
    source = old.replace_once(source,
        'raise RuntimeError("public E1 entity preflight failed: " + ", ".join(missing_names))',
        'raise MissingPublicEvidence(tuple(missing_names), tuple(perception_log))')
    source = old.replace_once(source,
        '        report["error"] = f"{type(exc).__name__}: {exc}"\n',
        '        report["error"] = f"{type(exc).__name__}: {exc}"\n'
        '        if isinstance(exc, MissingPublicEvidence):\n'
        '            report["initial_public_evidence_missing"] = exc.marker()\n')
    source = old.replace_once(source,
        '        report["resource_exhausted"] = resource_exhausted\n',
        '        if report.get("initial_public_evidence_missing"):\n'
        '            infrastructure_fault = initial_missing_has_other_fault(report,\n'
        '                action_system_fault=action_system_fault, commit_system_fault=commit_system_fault)\n'
        '        report["resource_exhausted"] = resource_exhausted\n')
    source = old.replace_once(source,
        '        elif raw.get("label") == "NOT_TRIGGERED":\n',
        '        elif report.get("initial_public_evidence_missing"):\n'
        '            label, qa = "NOT_TRIGGERED", "capability_failure"\n'
        '        elif raw.get("label") == "NOT_TRIGGERED":\n')
    return old.compiled(source, PACKAGE+'._compiled_runner', file)


def runner_module():
    # Compile just the actual old hook function; do not copy its runner,
    # alter its module globals, or patch code objects/constants.
    path = Path(old.__file__)
    expected = json.loads((HERE/'parent_hashes.json').read_text()) if source_custody_enabled() else None
    text = path.read_text()
    if source_custody_enabled() and hashlib.sha256(path.read_bytes()).hexdigest() != expected[str(path.relative_to(ROOT))]:
        raise RuntimeError('v1 hook parent changed')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'runner_module')
    function = ast.get_source_segment(text, node)
    function = old.replace_once(function,
        "return compiled(source, PACKAGE+'._compiled_runner', ROOT/path)",
        "return compiled_v2_runner(source, PACKAGE+'._compiled_runner', ROOT/path)")
    namespace = dict(vars(old), compiled_v2_runner=compile_runner)
    exec(compile(function, str(path), 'exec'), namespace)
    return namespace['runner_module']()
