"""One actual persistent local-builder hook; original Async rounds remain."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
import ast
import copy
from hashlib import sha256
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

from research.v4.candidates.multiround_bridge_20260924.driver import CodexPersistentSession
from research.v4.leash_v43.tool_boundary import inject_planner_policy, planner_turn_options

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = "S101-original-persistent-approved-policy-local-builder-optin-v1"
PARENT = ROOT / "research/v4/candidates/s1_user_revision_real_binding_20260928"
AUDIT = ROOT / "research/v4/candidates/s101_v43_multiround_consumer_seam_readonly_20260929"
FILES = ("__init__.py", "policy_transport.py", "contract.json", "check_policy.py", "HANDOFF.md")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sources():
    contract = json.loads((HERE / "contract.json").read_text())
    pins = {}
    for folder, field in ((PARENT, "parent_freeze_sha256"), (AUDIT, "audit_freeze_sha256")):
        frozen = folder / "source_freeze.json"
        if digest(frozen) != contract[field]:
            raise RuntimeError("qualified parent/audit source manifest identity differs")
        members = json.loads(frozen.read_text())
        for name, value in members.items():
            if name in pins and pins[name] != value:
                raise RuntimeError("qualified S101 parent/audit source conflict: " + name)
            if digest(ROOT / name) != value:
                raise RuntimeError("qualified S101 parent/audit source drift: " + name)
            pins[name] = value
        pins[str(frozen.relative_to(ROOT))] = digest(frozen)
    for folder, name in ((AUDIT, "FINAL_RECEIPT.json"), (AUDIT, "read_receipt.json")):
        path = folder / name
        pins[str(path.relative_to(ROOT))] = digest(path)
    for name in FILES:
        path = HERE / name
        pins[str(path.relative_to(ROOT))] = digest(path)
    sdk = Path(contract["installed_sdk_api_path"])
    if digest(sdk) != contract["installed_sdk_api_sha256"]:
        raise RuntimeError("installed Async SDK signature source differs")
    pins[str(sdk)] = digest(sdk)
    return dict(sorted(pins.items()))


def source_hashes(expected=None):
    if not source_custody_enabled():
        return {}
    frozen = json.loads((HERE / "source_freeze.json").read_text())
    if frozen["version"] != VERSION:
        raise ValueError("explicit persistent-policy source version differs")
    actual = sources()
    if actual != (frozen["sha256"] if expected is None else expected):
        raise RuntimeError("whole qualified persistent-policy source admission differs")
    return actual


def _approved_builder(original, method, policy, expected):
    def selected(*args, **kwargs):
        planner = original(*args, **kwargs)
        return inject_planner_policy(planner, method=method, approved_policy=policy,
                                     expected_policy_sha256=expected)
    return selected


def session_class(original_session_class=CodexPersistentSession, *, method,
                  approved_policy=None, expected_policy_sha256=None):
    """Root may bind this class before its existing DeferredPlannerSession."""
    source_hashes()
    if original_session_class is not CodexPersistentSession or method not in {"B0", "Ours", "B1", "B2", "B3"}:
        raise ValueError("original persistent class and an existing method profile required")
    if method == "B0":
        if approved_policy is not None or expected_policy_sha256 is not None:
            raise ValueError("B0 original prompt receives no policy prerequisite")
        return original_session_class
    # Use the existing validator before any builder/client/server construction.
    inject_planner_policy(SimpleNamespace(_turn_options={}), method=method,
                          approved_policy=approved_policy,
                          expected_policy_sha256=expected_policy_sha256)
    contract = json.loads((HERE / "contract.json").read_text()) if source_custody_enabled() else None
    original = original_session_class.__aenter__
    source = textwrap.dedent(inspect.getsource(original))
    if source_custody_enabled() and sha256(source.encode()).hexdigest() != contract["original_aenter_sha256"]:
        raise RuntimeError("original persistent enter source differs")
    tree = ast.parse(source)
    function = tree.body[0]
    imports = [node for node in function.body if isinstance(node, ast.ImportFrom)
               and node.module == "research.v3.bridge.b0"
               and [(x.name, x.asname) for x in node.names] == [("build_b0_planner", None)]]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "build_b0_planner"]
    if len(imports) != 1 or len(calls) != 1:
        raise RuntimeError("exact actual local persistent builder seam differs")
    if source_custody_enabled() and any(keyword.arg in {"model_id", "reasoning_effort"} for keyword in calls[0].keywords):
        raise RuntimeError("unexpected model override in original persistent builder")
    index = function.body.index(imports[0]) + 1
    alias = ast.parse("build_b0_planner = _v43_approved_builder(build_b0_planner)").body[0]
    function.body.insert(index, alias)
    if source_custody_enabled():
        reverted = copy.deepcopy(tree)
        del reverted.body[0].body[index]
        if ast.dump(reverted, include_attributes=False) != ast.dump(ast.parse(source), include_attributes=False):
            raise RuntimeError("local builder alias changed another session algorithm")
    namespace = dict(original.__globals__)
    namespace["_v43_approved_builder"] = lambda builder: _approved_builder(
        builder, method, approved_policy, expected_policy_sha256)
    exec(compile(ast.fix_missing_locations(tree), str(HERE / "policy_transport.py") + "::one-local-builder", "exec"), namespace)
    enter = namespace["__aenter__"]

    async def approved_enter(self):
        value = await enter(self)  # Original early client/thread failure stops its server.
        try:
            # Installed SDK accepts this trusted field at thread_start only.
            # Policy is now held by that one thread; original run_round stays
            # untouched and sends only its original legal per-turn kwargs.
            self._planner._turn_options = planner_turn_options(self._planner)
        except BaseException as exc:
            await original_session_class.__aexit__(self, type(exc), exc, exc.__traceback__)
            raise
        return value

    selected = type("ApprovedPolicyCodexPersistentSession", (original_session_class,), {
        "__aenter__": approved_enter, "__module__": __name__,
    })
    selected.original_session_class = original_session_class
    selected.bound_original_aenter = enter
    selected.policy_sha256 = expected_policy_sha256
    if source_custody_enabled():
        selected.selected_aenter_source = ast.unparse(tree)
        selected.original_aenter_sha256 = contract["original_aenter_sha256"]
        selected.local_builder_only_AST_change = True
        if (selected.run_round.__code__ is not original_session_class.run_round.__code__
                or selected.__aexit__.__code__ is not original_session_class.__aexit__.__code__):
            raise RuntimeError("original persistent round/close CodeType changed")
    return selected
