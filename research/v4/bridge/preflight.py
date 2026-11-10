"""Read-only acceptance validation and source freezing for accepted v4 R-task B0 smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = Path("research/v4/configs/instance_registry.json")
SUPPORTED = {"R1-01": "r1", "R2-01": "r2", "R4-01": "r4", "R5-01": "r5",
             "I1-01": "i1"}
BRIDGE_FILES = (
    "research/v4/bridge/__init__.py",
    "research/v4/bridge/preflight.py",
    "research/v4/bridge/protocol.py",
    "research/v4/bridge/observation.py",
    "research/v4/bridge/i1_integration.py",
    "research/v4/bridge/thread_owner.py",
    "research/v4/bridge/run_b0.py",
    "research/v3/bridge/b0.py",
    "research/v3/bridge/libero_env.py",
    "upstream/RPent-reproduce-libero/robots/libero/tools.py",
    "upstream/RPent-reproduce-libero/robots/libero/toolkit.py",
    "upstream/RPent-reproduce-libero/robots/libero/vla_server.py",
    "upstream/RPent-reproduce-libero/robots/libero/sam3_server.py",
    "upstream/RPent-reproduce-libero/rpent/planner/codex.py",
    "envs/libero/lib/python3.11/site-packages/robosuite/utils/binding_utils.py",
    "envs/libero/lib/python3.11/site-packages/robosuite/utils/camera_utils.py",
    "checkpoints/pi05-libero130/physical-intelligence/libero/norm_stats.json",
    "assets/manifest.csv",
)
MEMORY = Path("upstream/RPent-reproduce-libero/resources/libero/memory")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_accepted(instance_id: str, *, root: Path = ROOT) -> dict:
    """Require the accepted v4 base scene and unchanged task-local source."""
    if instance_id not in SUPPORTED:
        raise ValueError("B0 runner supports only registered accepted v4 bases")
    task = SUPPORTED[instance_id]
    registry = json.loads((root / REGISTRY).read_text())
    rows = [row for row in registry["instances"] if row["id"] == instance_id]
    if len(rows) != 1:
        raise ValueError(f"registry must contain exactly one {instance_id}")
    row = rows[0]
    if row.get("base_task") != task.upper() or not row.get("is_base_instance"):
        raise ValueError("registry row is not the expected v4 base task")
    if row.get("acceptance_status") != "accepted":
        raise ValueError("v4 base instance is not accepted")
    report_path = row.get("acceptance_report")
    if not isinstance(report_path, str):
        raise ValueError("registry acceptance report path is missing")
    relative_report = Path(report_path)
    expected_parent = Path(f"research/v4/tasks/{task}")
    if (relative_report.is_absolute() or ".." in relative_report.parts or
            relative_report.parent != expected_parent or
            not relative_report.name.startswith("acceptance-report") or
            relative_report.suffix != ".json"):
        raise ValueError("registry acceptance report path differs")
    report = json.loads((root / relative_report).read_text())
    if (report.get("instance_id") != instance_id or report.get("accepted") is not True
            or not str(report.get("protocol", "")).startswith(
                "SafeVLA-Harness v4 section 7.1")):
        raise ValueError("v4 section-7 acceptance is absent")
    instructions = row.get("instructions")
    if (not isinstance(instructions, list) or len(instructions) != 1 or
            instructions[0] != report.get("instruction")):
        raise ValueError("accepted instruction differs from current registry")
    prefix = f"research/v4/tasks/{task}/"
    required_names = (("template.py", "scene.py", "evaluator.py", "note.py",
                       "assets/I1-01/note.png", "assets/print-face.obj",
                       "assets/blunt_tool_visible_v1.xml") if task == "i1" else
                      ("assets.json", "template.py", "scene.py", "evaluator.py"))
    required = {f"{prefix}{name}" for name in required_names}
    reported = report.get("source_hashes")
    if not isinstance(reported, dict):
        raise ValueError("accepted report lacks source hashes")
    task_hashes = {path: sha for path, sha in reported.items()
                   if path.startswith(prefix)}
    if not required.issubset(task_hashes):
        raise ValueError("accepted report lacks required task-local hashes")
    # I1's accepted note, person, LIBERO objects and knife span many roots;
    # verifying only the task-local subset would not pin the reviewed scene.
    reviewed = ({path: sha for path, sha in reported.items()
                 if path != REGISTRY.as_posix()} if task == "i1" else task_hashes)
    for relative, expected in reviewed.items():
        source = Path(relative)
        if (source.is_absolute() or ".." in source.parts or not source.parts or
                not isinstance(expected, str) or len(expected) != 64 or
                any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError(f"invalid accepted source pin: {relative}")
        path = root / source
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"accepted source has changed: {relative}")
    # The report's old whole-registry hash is intentionally not compared: other
    # rows' administrative acceptance fields change as v4 progresses.
    return {"row": row, "report": report, "report_path": relative_report,
            "instruction": instructions[0], "task_hashes": task_hashes,
            "reviewed_hashes": reviewed}


def source_hashes(accepted: dict, *, root: Path = ROOT) -> dict[str, str]:
    paths = set(accepted["reviewed_hashes"])
    paths.update(BRIDGE_FILES)
    paths.update((REGISTRY.as_posix(), accepted["report_path"].as_posix()))
    # Freeze every Python module in the RPent checkout that this runner can
    # import, including RPC helpers. A local editable install may point at a
    # different checkout, so the runtime also validates module origins.
    for package in ("rpent", "robots/libero"):
        code_dir = root / "upstream/RPent-reproduce-libero" / package
        paths.update(path.relative_to(root).as_posix()
                     for path in code_dir.rglob("*.py"))
    memory_dir = root / MEMORY
    paths.update(path.relative_to(root).as_posix() for path in memory_dir.rglob("*.md"))
    result = {}
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = digest(path)
    return result


def freeze_sources(output: Path, hashes: dict[str, str], *, root: Path = ROOT) -> None:
    """Copy exact inputs outside agent-readable RPent output before startup."""
    for relative, expected in hashes.items():
        source = root / relative
        if digest(source) != expected:
            raise RuntimeError(f"source changed before freeze: {relative}")
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != expected:
            raise RuntimeError(f"source snapshot mismatch: {relative}")
    (output / "source-sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n")


def check_live_sources(hashes: dict[str, str], *, root: Path = ROOT) -> list[str]:
    return [relative for relative, expected in hashes.items()
            if not (root / relative).is_file() or digest(root / relative) != expected]


def check_cached_module_paths(*, root: Path = ROOT) -> None:
    """Reject same-named modules imported from another checkout before freeze."""
    prefixes = ("research.v3.bridge.", "research.v4.bridge.",
                "research.v4.tasks.r1.", "research.v4.tasks.r2.",
                "research.v4.tasks.r4.", "research.v4.tasks.r5.",
                "research.v4.tasks.i1.",
                "robots.libero.")
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefixes):
            continue
        origin = getattr(module, "__file__", None)
        if origin and root not in Path(origin).resolve().parents:
            raise RuntimeError(f"cached module from another checkout: {name}: {origin}")
