"""Select five accepted receiving/induction scenes in the existing REAL main."""
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace
import hashlib
import json
import marshal

from research.v4.candidates.s1_real_deferred_own_production_20260927 import factory as parent
from research.v4.candidates.user_rework_s1_induction_20260927 import scene_v4 as revision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
IDS = parent.IDS
VERSION = "S1-own-user-induction-receiving-deferred-real-optin-v1-20260928"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_path(instance):
    if instance not in IDS:
        raise ValueError("Only five registered S1 instances")
    return ROOT / "research/v4/results" / (
        instance.lower().replace("-", "_") +
        "_user_revision_v42_instance_acceptance_20260928.json")


def source_union():
    pins = dict(parent.source_hashes())
    pins[str((parent.HERE / "source_freeze.json").relative_to(ROOT))] = sha(
        parent.HERE / "source_freeze.json")
    for instance in IDS:
        path = receipt_path(instance)
        receipt = json.loads(path.read_text())
        if not receipt["accepted"] or receipt["instance"] != instance:
            raise RuntimeError("Current own S1 engineering acceptance missing")
        for name, digest in receipt["source_sha256"].items():
            actual_path = Path(name) if name.startswith("/") else ROOT / name
            if sha(actual_path) != digest:
                raise RuntimeError("Current accepted S1 source drift: " + name)
            if name in pins and pins[name] != digest:
                raise RuntimeError("Conflicting old/current physical-source pin: " + name)
            pins[name] = digest
        pins[str(path.relative_to(ROOT))] = sha(path)
    for path in HERE.glob("*.py"):
        pins[str(path.relative_to(ROOT))] = sha(path)
    return dict(sorted(pins.items()))


def admit(actual, expected):
    if expected is None or actual != expected:
        raise RuntimeError("Current S1 production source drift or incomplete freeze")
    return actual


def source_hashes():
    frozen = HERE / "source_freeze.json"
    expected = json.loads(frozen.read_text()) if frozen.exists() else None
    return admit(source_union(), expected)


@lru_cache(maxsize=5)
def selected_scene(instance):
    receipt_path(instance)
    return revision.factory(instance)


def replace_metadata_constant(code, old, new):
    """Only the accepted-receipt text changes; executable bytecode is retained."""
    count = 0
    values = []
    for value in code.co_consts:
        if isinstance(value, CodeType):
            value, changed = replace_metadata_constant(value, old, new)
            count += changed
        elif isinstance(value, str) and value == old:
            value = new
            count += 1
        values.append(value)
    return code.replace(co_consts=tuple(values)), count


def runner_module(instance):
    receipt_path(instance)
    template = parent.runner_module(instance)
    old_c, bridge, original_core = parent.components(instance)
    Scene = selected_scene(instance)
    accepted = str(receipt_path(instance).relative_to(ROOT))
    binding = replace(old_c.binding, acceptance=accepted)
    current_c = SimpleNamespace(**{**vars(old_c), "Scene": Scene, "binding": binding})
    core = parent.hooks.clone_module(original_core)
    # Assign explicit admission after clone_module has rebound local functions.
    core.source_hashes = source_hashes
    value = parent.hooks.clone_module(template)
    changed = 0
    for name, function in list(vars(value).items()):
        if isinstance(function, FunctionType) and function.__globals__ is vars(value):
            code, n = replace_metadata_constant(function.__code__, old_c.binding.acceptance, accepted)
            if n:
                rebound = FunctionType(code, vars(value), function.__name__, function.__defaults__, function.__closure__)
                rebound.__kwdefaults__ = function.__kwdefaults__
                rebound.__annotations__ = function.__annotations__
                rebound.__qualname__ = function.__qualname__
                value.__dict__[name] = rebound
                changed += n
    if changed != 1:
        raise RuntimeError("Exact current own acceptance metadata seam changed")
    value.S1StablePan300Facade = Scene
    value.run_s1_method_core = core.run_s1_method_core
    value.optin_core = core
    value.optin_bridge = bridge
    value.own_components = current_c
    value.source_hashes = source_hashes

    def preflight(args, **kwargs):
        output, _ = template.preflight(args, **kwargs)
        return output, source_hashes()

    value.preflight = preflight
    value.PROTOCOL = template.PROTOCOL + "-" + VERSION
    value.parent_runner = template
    value.current_scene_acceptance = accepted
    value.original_compiled_main_sha256 = template.compiled_main_sha256
    value.current_main_code_sha256 = hashlib.sha256(marshal.dumps(value.main.__code__)).hexdigest()
    value.acceptance_metadata_replacements = changed
    value.native_method_ready = False
    value.launch_authorized = False
    return value
