"""Small per-run manifest. Source custody is disabled by default."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import time

SOURCE_FILES = (
    "research/v4/leash_v43/adapter.py",
    "research/v4/leash_v43/dispatch.py",
    "research/v4/leash_v43/registry.py",
    "research/v4/leash_v43/method_inputs.py",
    "research/v4/leash_v43/tool_boundary.py",
    "research/v4/leash_v43/reviewer_transport.py",
    "research/v4/candidates/leash_reviewer_transport_20260924/transport.py",
    "research/v4/leash_v43/public_access.py",
    "research/v4/leash_v43/run_manifest.py",
    "research/v4/bridge/thread_owner.py",
    "research/v3/bridge/b0.py",
    "research/v3/bridge/libero_env.py",
)

# Current shared configurations plus the isolated planner transport use a finite
# bounded list, including the current family-specific source pins.
# This remains a finite hash list; it does not collect source or traverse imports.
MAX_SOURCE_FILES = 128


def source_custody_enabled():
    value = os.environ.get("SAFEVLA_SOURCE_CUSTODY", "0")
    if value not in {"0", "1"}:
        raise ValueError("SAFEVLA_SOURCE_CUSTODY must be 0 or 1")
    return value == "1"


def atomic_json(path, value, *, max_output_bytes=None, budget_directory=None):
    """Write a complete ordinary JSON file atomically, then its SHA-256."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if max_output_bytes is not None:
        directory=Path(budget_directory) if budget_directory is not None else path.parent
        current=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        # Include the atomic temporary file while the original still exists.
        if current+len(data)+65>max_output_bytes:
            raise RuntimeError('Output byte limit would be exceeded by '+path.name)
    temporary.write_bytes(data)
    temporary.replace(path)
    path.with_name(path.name + ".sha256").write_text(sha256(data).hexdigest() + "\n")


def _hashes(repo, files):
    return {name: sha256((repo / name).read_bytes()).hexdigest() for name in files}


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)


def _cli_metadata(executable):
    """A finite local version probe is telemetry, not model admission.

    subprocess.run kills and reaps a timed-out probe. Do not retry it or invent
    a version; the actual configured SDK process still has to work normally.
    Other startup errors retain their original exception behavior.
    """
    try:
        reply = subprocess.run([executable, '--version'], capture_output=True,
                               text=True, timeout=10)
    except subprocess.TimeoutExpired as error:
        return dict(path=executable, version=None, exit_code=None,
                    probe_status='timeout', probe_timeout_s=error.timeout,
                    probe_error='TimeoutExpired')
    return dict(path=executable, version=reply.stdout.strip(),
                exit_code=reply.returncode, probe_status='completed')


@contextmanager
def run_manifest(output, *, repo, method, instance_id, seed, gates,
                 planner_model, planner_reasoning, reviewer_model, reviewer_reasoning,
                 source_files=SOURCE_FILES, run_type="lite", budgets=None,
                 familiarization=None, recording=None, policy=None):
    """Rehash the same finite list in finally, including exceptional exits.

    The git diff is a separate file, scoped to the same explicit source list.
    Unrelated workspace results and status files are not collected. No source,
    AST, callable, or object graph is stored in JSON. Callers report invalid drift.
    """
    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = tuple(dict.fromkeys(source_files))
    if not files or len(files) > MAX_SOURCE_FILES:
        raise ValueError(f"run manifest requires a finite source list of at most {MAX_SOURCE_FILES} files")
    for name in files:
        if not (repo / name).resolve().is_relative_to(repo):
            raise ValueError("manifest source must be inside the repository")
    started = time.monotonic()
    overhead_start = time.monotonic()
    before = _hashes(repo, files)
    packages = {}
    for name in ("mujoco", "robosuite", "torch", "numpy", "codex", "openai-codex", "codex-cli-bin"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    # Literal paths prevent a source filename from becoming a Git glob. Compare
    # with HEAD so staged and unstaged source edits are both retained, without
    # copying unrelated (potentially enormous) result/status changes every run.
    source_paths = tuple(dict.fromkeys(os.path.relpath(repo / name, repo) for name in files))
    pathspecs = [":(literal)" + name for name in source_paths]
    diff = _git(repo, "diff", "--binary", "--no-ext-diff", "--no-textconv",
                "HEAD", "--", *pathspecs)
    # New runtime files may be untracked: retain their actual diff as well.
    for name, pathspec in zip(source_paths, pathspecs):
        tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", pathspec],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not tracked:
            part = subprocess.run(["git", "-C", str(repo), "diff", "--no-index",
                                   "--binary", "--no-ext-diff", "--no-textconv",
                                   "--", "/dev/null", name], capture_output=True)
            # git diff uses 1 for an actual difference; other failures must not
            # silently discard the only saved patch for an untracked source.
            if part.returncode not in (0, 1):
                raise subprocess.CalledProcessError(part.returncode, part.args,
                                                    output=part.stdout, stderr=part.stderr)
            diff += part.stdout
    if diff:
        (output / "working_tree.diff").write_bytes(diff)
    cli = None
    if os.environ.get('CODEX_BIN'):
        cli = _cli_metadata(os.environ['CODEX_BIN'])
    manifest = {
        "run_type": run_type,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(repo, "rev-parse", "HEAD").decode().strip(),
        "dirty_diff": "working_tree.diff" if diff else None,
        "diff_scope": "explicit_source_files_only",
        "diff_included_source_paths": list(source_paths),
        "python": platform.python_version(), "packages": packages,
        "codex_cli": cli,
        "method": method, "instance_id": instance_id, "seed": seed, "gates": gates,
        "planner": {"model": planner_model, "reasoning": planner_reasoning},
        "reviewer": {"model": reviewer_model, "reasoning": reviewer_reasoning},
        "budgets": budgets,
        "familiarization": familiarization,
        "recording": recording,
        "policy": policy,
        "source_custody": source_custody_enabled(),
        "source_sha256_start": before,
        "source_file_limit": MAX_SOURCE_FILES,
        "valid": None,
    }
    atomic_json(output / "run_manifest.json", manifest)
    overhead = time.monotonic() - overhead_start
    failure = None
    try:
        yield manifest
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        overhead_start = time.monotonic()
        try:
            after = _hashes(repo, files)
            drift = [name for name in files if after[name] != before[name]]
        except OSError as exc:
            after, drift = {}, [f"source read failed: {exc}"]
        manifest.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                        source_sha256_end=after, source_drift=drift,
                        valid=not drift, error=failure)
        overhead += time.monotonic() - overhead_start
        manifest["elapsed_s"] = time.monotonic() - started
        manifest["manifest_overhead_s"] = overhead
        manifest["manifest_overhead_fraction"] = overhead / max(manifest["elapsed_s"], 1e-9)
        atomic_json(output / "run_manifest.json", manifest)
        if drift and failure is None:
            raise RuntimeError(f"run invalid: finite source list changed: {drift}")


def write_gate_decision(path, *, gate, trigger, candidate, epoch, decision, reason,
                        reviewer_called, elapsed_s, dispatched, invalidated):
    row = dict(gate=gate, trigger=trigger, candidate=candidate, epoch=epoch,
               decision=decision, reason=reason, reviewer_called=reviewer_called,
               elapsed_s=elapsed_s, dispatched=dispatched, invalidated=invalidated)
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_gate(*, gate, trigger, candidate, epoch, decision, reason, reviewer_called,
                elapsed_s=0., dispatched=False, invalidated=False):
    """Ordinary decision log; inactive outside an actual episode output."""
    output = os.environ.get("SAFEVLA_RUN_OUTPUT")
    if output:
        write_gate_decision(Path(output) / "decisions.jsonl", gate=gate, trigger=trigger,
                            candidate=candidate, epoch=epoch, decision=decision, reason=reason,
                            reviewer_called=reviewer_called, elapsed_s=elapsed_s,
                            dispatched=dispatched, invalidated=invalidated)
