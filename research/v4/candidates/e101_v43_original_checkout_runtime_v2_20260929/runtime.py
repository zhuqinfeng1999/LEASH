"""Bind the original RPent source before compiling the immutable E101 donor.

Only source selection occurs here.  The original binder rejects foreign cached
modules.  Its context is restored, and the provider later binds the reviewed
typed decoder to the module of the actual, already supplied SAM client.
"""
from contextlib import contextmanager
from hashlib import sha256
import importlib
import json
import marshal
from pathlib import Path
import sys
import tempfile
from types import FunctionType, ModuleType, SimpleNamespace

from research.v4.bridge import run_b0 as original_loader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DONOR = ROOT / "research/v4/candidates/e101_v43_consumer_runtime_20260929"
VERSION = "E101-own-v43-original-RPent-source-context-runtime-v2"
HELPER = "research.v4.candidates.r4_02_typed_public_masks_20260927.public"
SAMPLER = "rpent.utils.sam3_client"
FILES = ("__init__.py", "runtime.py", "contract.json", "HANDOFF.md", "check_source.py")
Consumer = None


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sources():
    freeze = DONOR / "source_freeze.json"
    pins = dict(json.loads(freeze.read_text())["sha256"])
    pins[str(freeze.relative_to(ROOT))] = digest(freeze)
    for name in ("rpent/__init__.py", "rpent/utils/__init__.py", "rpent/utils/rpc.py",
                 "rpent/utils/sam3_client.py"):
        path = original_loader.RPENT / name
        key = str(path.relative_to(ROOT))
        actual = digest(path)
        if key in pins and pins[key] != actual:
            raise RuntimeError("Original RPent source conflicts with immutable donor: " + key)
        pins[key] = actual
    for name in FILES:
        path = HERE / name
        pins[str(path.relative_to(ROOT))] = digest(path)
    for key, expected in pins.items():
        path = Path(key) if key.startswith("/") else ROOT / key
        if digest(path) != expected:
            raise RuntimeError("Complete original-RPent/E101 source changed: " + key)
    return dict(sorted(pins.items()))


def source_hashes(expected=None):
    freeze = json.loads((HERE / "source_freeze.json").read_text())
    if freeze["version"] != VERSION:
        raise ValueError("Exact independent E101 source context version required")
    pins = sources()
    if pins != (freeze["sha256"] if expected is None else expected):
        raise RuntimeError("Whole E101 context source membership/bytes changed")
    return pins


def _relevant(name):
    return (name == "rpent" or name.startswith("rpent.") or name == "robots.libero"
            or name.startswith("robots.libero.") or name == HELPER)


def _module_rows():
    return {name: {"id": id(module), "file": getattr(module, "__file__", None)}
            for name, module in sorted(sys.modules.items()) if _relevant(name)}


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _one_code(code, name):
    matches = [value for value in code.co_consts if isinstance(value, type(code)) and value.co_name == name]
    if len(matches) != 1:
        raise RuntimeError("Exact original emitted callable missing/duplicated: " + name)
    return matches[0]


@contextmanager
def original_checkout(evidence):
    """Run the original strict binder, retaining and restoring actual objects."""
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    old_path = list(sys.path)
    old_modules = {name: module for name, module in sys.modules.items() if _relevant(name)}
    # Restore package attributes too: import machinery attaches loaded children.
    attributes = [(module, dict(vars(module))) for name, module in sys.modules.items()
                  if name in {"rpent", "rpent.utils", "robots", "robots.libero", HELPER.rsplit(".", 1)[0]}]
    _write(evidence / "before_binder.json", {"sys_path": old_path, "modules": _module_rows(),
        "binder_id": id(original_loader.bind_rpent_checkout),
        "binder_code_id": id(original_loader.bind_rpent_checkout.__code__),
        "binder_file": original_loader.bind_rpent_checkout.__code__.co_filename,
        "binder_marshal_sha256": sha256(marshal.dumps(original_loader.bind_rpent_checkout.__code__)).hexdigest(),
        "checkout": str(original_loader.RPENT.resolve())})
    try:
        source_hashes()
        path = Path(original_loader.__file__).resolve()
        emitted = compile(path.read_text(), str(path), "exec", dont_inherit=True)
        declared_binder = _one_code(emitted, "bind_rpent_checkout")
        _write(evidence / "binder_emitted_before_guard.json", {
            "actual_code_id": id(original_loader.bind_rpent_checkout.__code__),
            "emitted_code_id": id(declared_binder),
            "actual_marshal_sha256": sha256(marshal.dumps(original_loader.bind_rpent_checkout.__code__)).hexdigest(),
            "emitted_marshal_sha256": sha256(marshal.dumps(declared_binder)).hexdigest(),
            "strict_code_equal": original_loader.bind_rpent_checkout.__code__ == declared_binder})
        if original_loader.bind_rpent_checkout.__code__ != declared_binder:
            raise RuntimeError("Original strict checkout binder CodeType changed")
        original_loader.bind_rpent_checkout()
        _write(evidence / "after_binder_before_helper.json", {"sys_path": list(sys.path), "modules": _module_rows()})
        yield
    finally:
        for name in tuple(sys.modules):
            if _relevant(name) and name not in old_modules:
                del sys.modules[name]
        sys.modules.update(old_modules)
        for module, previous in attributes:
            for key in tuple(vars(module)):
                if key not in previous:
                    delattr(module, key)
            vars(module).update(previous)
        sys.path[:] = old_path
        importlib.invalidate_caches()
        restored = ({name: module for name, module in sys.modules.items() if _relevant(name)} == old_modules
                    and sys.path == old_path
                    and all(vars(module) == previous for module, previous in attributes))
        _write(evidence / "restored.json", {"sys_path": list(sys.path), "modules": _module_rows(),
                                            "same_original_objects": restored})
        if not restored:
            raise RuntimeError("Original RPent source scope failed to restore")


def _sampler_receipt(module):
    methods = ("__init__", "segment", "_decode_result")
    return {"module_id": id(module), "file": getattr(module, "__file__", None),
        "client_id": id(getattr(module, "Sam3Client", None)),
        "result_id": id(getattr(module, "Sam3Result", None)),
        "methods": {name: {"function_id": id(getattr(module.Sam3Client, name)),
            "code_id": id(getattr(module.Sam3Client, name).__code__),
            "code_sha256": sha256(marshal.dumps(getattr(module.Sam3Client, name).__code__)).hexdigest()}
            for name in methods},
        "decoder_result_is_module_result": module.Sam3Client._decode_result.__globals__.get("Sam3Result") is module.Sam3Result}


def _require_sampler(module, reference):
    path = Path(module.__file__).resolve()
    expected = (original_loader.RPENT / "rpent/utils/sam3_client.py").resolve()
    if module.__name__ != SAMPLER or path != expected:
        raise RuntimeError("Actual SAM client is outside the original admitted checkout")
    source_hashes()
    if (module.Sam3Client.__module__ != SAMPLER or module.Sam3Result.__module__ != SAMPLER
            or module.Sam3Client._decode_result.__globals__.get("Sam3Result") is not module.Sam3Result
            or any(getattr(module.Sam3Client, name).__code__ != reference[name]
                   for name in ("__init__", "segment", "_decode_result"))):
        raise RuntimeError("Actual original SAM class/decoder source identity changed")
    return module


def _function(function, namespace, *, closure=None):
    selected = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__,
                            function.__closure__ if closure is None else closure)
    selected.__kwdefaults__ = function.__kwdefaults__
    return selected


def _cell(value):
    return (lambda: value).__closure__[0]


def _typed_class(donor, module, reference):
    """Reuse both reviewed segment CodeTypes with the actual decoder result."""
    _require_sampler(module, reference)
    old_client = donor.Client
    namespace = dict(old_client.segment.__globals__, Sam3Result=module.Sam3Result)
    client = type("E101ActualOriginalTypedClient", (old_client,), {
        "__init__": old_client.__init__, "segment": _function(old_client.segment, namespace)})
    old = donor._TypedClient
    if old.segment.__code__.co_freevars != ("__class__",):
        raise RuntimeError("Original E101 typed-client closure source changed")
    typed = type("E101ActualOriginalMaskClient", (client,), {})
    typed.segment = _function(old.segment, dict(old.segment.__globals__), closure=(_cell(typed),))
    return typed


def _provider(donor, reference, components, scene, adapter, sam_client):
    """Bind the already supplied original client before the unchanged provider."""
    module = sys.modules.get(type(sam_client).__module__)
    if module is None or type(sam_client) is not getattr(module, "Sam3Client", None):
        raise RuntimeError("An actual admitted original Sam3Client instance is required")
    typed = _typed_class(donor, module, reference)

    def reader_class(selected_components):
        reader = donor.public_reader_class(selected_components)
        box = _function(donor._box, dict(donor._box.__globals__, _TypedClient=typed))
        # This is the generated own reader's actual query global, not an unused alias.
        reader.__call__ = _function(reader.__call__, dict(reader.__call__.__globals__, _box=box))
        return reader

    original = donor.public_provider
    selected = _function(original, dict(original.__globals__, public_reader_class=reader_class))
    return selected(components, scene, adapter, sam_client)


def namespace(*, evidence_dir=None):
    global Consumer
    admitted = source_hashes()
    evidence = Path(evidence_dir) if evidence_dir else Path(tempfile.mkdtemp(prefix="source_selection_", dir=HERE))
    if evidence.resolve().parent != HERE or evidence.exists() and any(evidence.iterdir()):
        raise ValueError("Each E101 selection requires an empty independent candidate evidence directory")
    with original_checkout(evidence):
        # The real original package is selected before this reviewed import.
        sampler = importlib.import_module(SAMPLER)
        _write(evidence / "sampler_before_guard.json", _sampler_receipt(sampler))
        path = Path(sampler.__file__).resolve()
        sampler_emitted = compile(path.read_text(), str(path), "exec", dont_inherit=True)
        class_emitted = _one_code(sampler_emitted, "Sam3Client")
        reference = {name: _one_code(class_emitted, name)
                     for name in ("__init__", "segment", "_decode_result")}
        _write(evidence / "sampler_emitted_before_guard.json", {
            name: {"emitted_code_id": id(code), "actual_code_id": id(getattr(sampler.Sam3Client, name).__code__),
                "emitted_sha256": sha256(marshal.dumps(code)).hexdigest(),
                "strict_code_equal": code == getattr(sampler.Sam3Client, name).__code__}
            for name, code in reference.items()})
        _require_sampler(sampler, reference)
        helper = importlib.import_module(HELPER)
        _write(evidence / "helper_before_guard.json", {"module_id": id(helper), "file": helper.__file__,
            "client_id": id(helper.Client), "result_id": id(helper.Sam3Result),
            "same_actual_result": helper.Sam3Result is sampler.Sam3Result})
        if helper.Sam3Result is not sampler.Sam3Result:
            raise RuntimeError("Reviewed helper is not bound to this actual original SAM result")
        # Compile the immutable donor into a new private namespace, never import
        # the failed donor module or execute its old checker.
        donor = ModuleType(__name__ + "._private_donor")
        donor.__dict__.update(__file__=str(DONOR / "runtime.py"), __package__=__package__)
        emitted = compile((DONOR / "runtime.py").read_text(), str(DONOR / "runtime.py"), "exec", dont_inherit=True)
        exec(emitted, donor.__dict__)
        original_provider = donor.public_provider

        def actual_provider(components, scene, adapter, sam_client):
            return _provider(donor, reference, components, scene, adapter, sam_client)

        donor.public_provider = actual_provider
        donor.source_hashes = source_hashes
        Consumer = donor.Consumer
        selected = donor.namespace()
        selected.source_hashes = source_hashes
        selected.source_admission = admitted
        selected.source_context_version = VERSION
        selected.source_context = SimpleNamespace(donor=donor, reference=reference,
            sampler=sampler, helper=helper, original_provider=original_provider, evidence_dir=evidence,
            emitted=emitted)
        _write(evidence / "private_compile_identity.json", {"module_id": id(donor),
            "Consumer_id": id(Consumer), "selected_consumer_same": selected.Consumer is Consumer,
            "actual_constructor_provider": donor.E101BaseConsumer.__init__.__globals__["public_provider"] is actual_provider,
            "actual_guard": donor.namespace.__globals__["source_hashes"] is source_hashes,
            "original_provider_code_id": id(original_provider.__code__),
            "emitted_sha256": sha256(marshal.dumps(emitted)).hexdigest(),
            "runtime_qualified": False, "native_ready": False, "launch_authorized": False})
    source_hashes()
    return selected
