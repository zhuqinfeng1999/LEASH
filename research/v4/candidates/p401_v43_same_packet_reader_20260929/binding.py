"""Capture the actual own factory emission into this independent scope."""
import importlib
import json
import marshal
from hashlib import sha256
from pathlib import Path
from types import FunctionType, SimpleNamespace

from research.v4.candidates.e101_v43_original_checkout_runtime_v2_20260929 import runtime as context
from . import public

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = "P401-own-full-corner-same-packet-typed-reader-source-v1"
PARENTS = ("user_revision_real_p4_v4_20260928", "p401_current_outlet_hd_20260928",
           "e101_v43_original_checkout_runtime_v2_20260929", "r202_v43_public_fault_types_20260929")
FILES = ("__init__.py", "binding.py", "public.py", "contract.json", "HANDOFF.md", "check_source.py")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sources():
    pins = {}
    for name in PARENTS:
        path = ROOT / "research/v4/candidates" / name / "source_freeze.json"
        value = json.loads(path.read_text())
        mapping = value["sha256"] if "sha256" in value else value
        for key, expected in mapping.items():
            if key in pins and pins[key] != expected:
                raise RuntimeError("Complete immutable own source obligations conflict: " + key)
            pins[key] = expected
        pins[str(path.relative_to(ROOT))] = digest(path)
    for name in FILES:
        path = HERE / name
        pins[str(path.relative_to(ROOT))] = digest(path)
    for key, expected in pins.items():
        if digest(Path(key) if key.startswith("/") else ROOT / key) != expected:
            raise RuntimeError("Whole P401 reader source changed: " + key)
    return dict(sorted(pins.items()))


def source_hashes(expected=None):
    frozen = json.loads((HERE / "source_freeze.json").read_text())
    if frozen["version"] != VERSION:
        raise ValueError("Exact independent P401 reader version required")
    actual = sources()
    if actual != (frozen["sha256"] if expected is None else expected):
        raise RuntimeError("Whole P401 source membership/bytes changed")
    return actual


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def code_row(function):
    return {"function_id": id(function), "code_id": id(function.__code__),
            "filename": function.__code__.co_filename,
            "marshal_sha256": sha256(marshal.dumps(function.__code__)).hexdigest()}


def select(*, evidence_dir):
    admission = source_hashes()
    evidence = Path(evidence_dir)
    if evidence.resolve().parent != HERE or evidence.exists() and any(evidence.iterdir()):
        raise ValueError("Each selection requires an empty private candidate output")
    evidence.mkdir(parents=True, exist_ok=True)
    with context.original_checkout(evidence / "original_checkout"):
        original = importlib.import_module("research.v4.candidates.user_revision_real_p4_v4_20260928.binding")
        sampler = importlib.import_module(context.SAMPLER)
        helper = importlib.import_module(context.HELPER)
        typed = importlib.import_module("research.v4.candidates.r202_v43_public_fault_types_20260929.public")
        captured = []
        modules_before = dict(__import__("sys").modules)
        emitted_modules = []
        compiled = original.core.compiled
        def capture(source, name, file, supplied=None):
            module = compiled(source, name, file, supplied)
            emitted_modules.append(name)
            if Path(file).name == "public_boundary.py":
                captured.append((source, module, file))
            return module
        factory = original.core.components.__wrapped__
        producer = public.function(factory, dict(factory.__globals__, compiled=capture))
        facade = SimpleNamespace(components=SimpleNamespace(__wrapped__=producer))
        selector = public.function(original.components.__wrapped__,
                                   dict(original.components.__wrapped__.__globals__, core=facade))
        try:
            components = selector("P4-01")
            if len(captured) != 1:
                raise RuntimeError("Exact own public-module emission was not captured once")
            source, module, file = captured[0]
            (evidence / "original_public_module.py.txt").write_text(source)
            emitted = compile(source, str(file), "exec", dont_inherit=True)
            cls_code = context._one_code(emitted, "PublicWorkspaceReader")
            acquisition = components.PublicWorkspaceReader.__call__
            expected = context._one_code(cls_code, "__call__")
            original_geometry = {name: getattr(components.P4ActionBoundary, name)
                                 for name in ("_proposal", "execute_tool", "__init__")}
            geometry_code = context._one_code(emitted, "P4ActionBoundary")
            comparisons = {name: fn.__code__ == context._one_code(geometry_code, name)
                           for name, fn in original_geometry.items()}
            write(evidence / "actual_emitted_before_guard.json", {
                "acquisition": code_row(acquisition), "emitted_code_id": id(expected),
                "acquisition_strict_equal": acquisition.__code__ == expected,
                "geometry": {name: code_row(fn) for name, fn in original_geometry.items()},
                "geometry_strict_equal": comparisons,
                "configuration_box": code_row(module.configuration_box),
                "configuration_strict_equal": module.configuration_box.__code__ == context._one_code(emitted, "configuration_box"),
                "producer_code_unchanged": producer.__code__ is factory.__code__,
                "selector_code_unchanged": selector.__code__ is original.components.__wrapped__.__code__,
                "binding": vars(components.binding), "Scene": components.Scene.__module__,
                "scorer": components.Evaluator.__module__, "sampler": context._sampler_receipt(sampler),
                "helper_result_same_actual": helper.Sam3Result is sampler.Sam3Result})
            if (acquisition.__code__ != expected or not all(comparisons.values())
                    or module.configuration_box.__code__ != context._one_code(emitted, "configuration_box")
                    or helper.Sam3Result is not sampler.Sam3Result):
                raise RuntimeError("Actual original emitted own reader/geometry/decoder identity changed")
            reader = public.reader_class(components.PublicWorkspaceReader, source,
                client=helper.Client, result_class=sampler.Sam3Result, detection_miss=helper.DetectionMiss,
                capture=typed._capture, fault=typed._fault, guard=source_hashes)
            (evidence / "selected_acquisition.py.txt").write_text(reader.selected_acquisition_source)
            selected = SimpleNamespace(PublicWorkspaceReader=reader, original_components=components,
                original_module=module, source_admission=admission, source_hashes=source_hashes,
                original_sampler=sampler, original_helper=helper, original_typed=typed,
                runtime_qualified=False, native_ready=False, launch_authorized=False,
                policy=None, calibration=None, grasp_assumption="unverified", vla_chunk_controls=None)
            write(evidence / "selected_before_guard.json", {
                "reader_id": id(reader), "call": code_row(reader.__call__),
                "constructor_code_unchanged": reader.__init__.__code__ is components.PublicWorkspaceReader.__init__.__code__,
                "locator_code_unchanged": reader.locate_vla_target.__code__ is components.PublicWorkspaceReader.locate_vla_target.__code__,
                "capture_code_unchanged": reader.__call__.__globals__["_capture"].__code__ is typed._capture.__code__,
                "actual_complete_guard": reader.__call__.__globals__["_whole_source_guard"] is source_hashes,
                "inverse_ast_exact": reader.inverse_ast_exact, "changes": reader.source_changes})
        finally:
            import sys
            for name in emitted_modules:
                if name in modules_before:
                    sys.modules[name] = modules_before[name]
                else:
                    sys.modules.pop(name, None)
            write(evidence / "emission_restore.json", {"exact_original_emitted_modules": all(
                sys.modules.get(name) is modules_before.get(name) for name in emitted_modules)})
    source_hashes()
    return selected
