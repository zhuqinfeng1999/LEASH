"""Preserve the qualified own reader while binding its supplied SAM decoder.

Namespace selection is source-only.  The provider later verifies the actual
already supplied client before constructing the original reader; it never
creates an SDK client or changes the public geometry algorithm.
"""
import json
import marshal
from hashlib import sha256
from pathlib import Path
import sys
from types import SimpleNamespace

from research.v4.candidates.p401_v43_same_packet_reader_20260929 import binding as parent
from research.v4.candidates.p401_v43_same_packet_reader_20260929 import public
from research.v4.candidates.e101_v43_original_checkout_runtime_v2_20260929 import runtime as context

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = "P401-supplied-original-client-provider-source-v1"
FILES = ("__init__.py", "binding.py", "check_source.py", "contract.json", "HANDOFF.md", "SOURCE_COMMANDS.md")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sources():
    contract = json.loads((HERE / "contract.json").read_text())
    pins = {}
    for name, expected in contract["parent_manifests"].items():
        path = ROOT / name
        if digest(path) != expected:
            raise RuntimeError("Exact immutable parent manifest changed: " + name)
        value = json.loads(path.read_text())
        mapping = value["sha256"] if "sha256" in value else value
        for key, member in mapping.items():
            if key in pins and pins[key] != member:
                raise RuntimeError("Complete parent source obligations conflict: " + key)
            pins[key] = member
        pins[name] = expected
    for name, expected in contract["qualified_receipts"].items():
        if name in pins and pins[name] != expected:
            raise RuntimeError("Qualified source receipt conflicts: " + name)
        pins[name] = expected
    for name in FILES:
        path = HERE / name
        pins[str(path.relative_to(ROOT))] = digest(path)
    for key, expected in pins.items():
        path = Path(key) if key.startswith("/") else ROOT / key
        if digest(path) != expected:
            raise RuntimeError("Whole P401 supplied-client source changed: " + key)
    return dict(sorted(pins.items()))


# The already-qualified complete-membership guard CodeType is retained.  Only
# this independent scope's version, HERE and source-map producer are selected.
source_hashes = public.function(parent.source_hashes, dict(
    parent.source_hashes.__globals__, HERE=HERE, VERSION=VERSION, sources=sources))


def _reader_for_module(selected, reference, module):
    """Bind class factories only; no reader, client, result or capture call."""
    source_hashes()
    context._require_sampler(module, reference)
    client = selected.original_helper.Client
    typed = type("P401SuppliedOriginalTypedClient", (client,), {
        "__init__": client.__init__,
        "segment": public.function(client.segment, dict(
            client.segment.__globals__, Sam3Result=module.Sam3Result))})
    original = selected.PublicWorkspaceReader
    segment = original.__call__.__globals__["_segment"]
    rebound_segment = public.function(segment, dict(segment.__globals__, _selected_Client=typed))
    acquisition = public.function(original.__call__, dict(
        original.__call__.__globals__, _segment=rebound_segment))
    return type("P401SuppliedOriginalClientReader", (original,), {"__call__": acquisition})


def _supplied_reader(selected, reference, sam_client):
    module = sys.modules.get(type(sam_client).__module__)
    if module is None or type(sam_client) is not getattr(module, "Sam3Client", None):
        raise RuntimeError("An actual admitted original Sam3Client instance is required")
    return _reader_for_module(selected, reference, module)


def namespace(*, evidence_dir):
    admission = source_hashes()
    evidence = Path(evidence_dir)
    # Reuse the exact own source selector.  Both its actual acquisition guard
    # and fresh diagnostic outputs consume this independent complete scope.
    selector = public.function(parent.select, dict(
        parent.select.__globals__, HERE=HERE, source_hashes=source_hashes))
    selected = selector(evidence_dir=evidence)
    sampler = selected.original_sampler
    path = Path(sampler.__file__).resolve()
    emitted = compile(path.read_text(), str(path), "exec", dont_inherit=True)
    cls = context._one_code(emitted, "Sam3Client")
    reference = {name: context._one_code(cls, name)
                 for name in ("__init__", "segment", "_decode_result")}
    parent.write(evidence / "supplied_decoder_emitted_before_guard.json", {
        "module": context._sampler_receipt(sampler),
        "emitted": {name: {"code_id": id(code),
            "marshal_sha256": sha256(marshal.dumps(code)).hexdigest(),
            "strict_equal": code == getattr(sampler.Sam3Client, name).__code__}
            for name, code in reference.items()},
        "selector_CodeType_same": selector.__code__ is parent.select.__code__,
        "guard_CodeType_same": source_hashes.__code__ is parent.source_hashes.__code__})
    context._require_sampler(sampler, reference)

    def public_provider(scene, adapter, sam_client):
        reader = _supplied_reader(selected, reference, sam_client)
        return reader(scene, adapter, sam_client)

    original = selected.original_module
    result = SimpleNamespace(
        original_components=selected.original_components, original_module=original,
        PublicWorkspaceReader=selected.PublicWorkspaceReader, public_provider=public_provider,
        configuration_box=original.configuration_box, bind_name=original.bind_name,
        bind_task=original.bind_task, P4ActionBoundary=selected.original_components.P4ActionBoundary,
        original_sampler=sampler, original_helper=selected.original_helper,
        decoder_reference=reference, selected_reader=selected,
        source_hashes=source_hashes, source_admission=admission,
        source_context_version=VERSION, evidence_dir=evidence,
        policy=None, calibration=None, grasp_assumption="unverified", vla_chunk_controls=None,
        runtime_qualified=False, native_ready=False, launch_authorized=False)
    source_hashes()
    return result
