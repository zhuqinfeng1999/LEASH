"""Fail-closed official PreToolUse handler, with an explicit scoped binding.

Only the public ``toolCall.name`` / ``toolCall.args`` contract is consumed.
Explicit fully scoped names include the actual CLI's single-underscore form.
Any different representation is denied until the host verifies its format.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any
import uuid


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    # A bounded exclusive lock file also works on the receiving Windows host.
    # A stale lock fails closed instead of risking an unaudited execution.
    lock_path = path.with_name(path.name + ".append.lock")
    deadline = time.monotonic() + 3
    while True:
        try:
            lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Qualification audit lock unavailable")
            time.sleep(.01)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
    finally:
        os.close(lock_fd)
        lock_path.unlink()


def valid_arguments(tool: str, arguments: Any) -> bool:
    if not isinstance(arguments, dict):
        return False
    if tool == "observe_test":
        return not arguments
    if tool == "record_test":
        return (set(arguments) == {"token"} and isinstance(arguments["token"], str)
                and 1 <= len(arguments["token"]) <= 32)
    return False


def hook_arguments(arguments: Any) -> dict[str, Any] | None:
    """Remove only the provider UI field observed in the real high-effort run.

    ``toolSummary`` is not tool authority, server scope, or a tool argument.
    Keep the original payload intact in the audit and validate a copy only.
    """
    if not isinstance(arguments, dict):
        return None
    copied = dict(arguments)
    if "toolSummary" in copied:
        summary = copied.pop("toolSummary")
        if not isinstance(summary, str) or not 1 <= len(summary) <= 256 or "\x00" in summary:
            return None
    return copied


def inspect_image_load(payload: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any] | None]:
    denied = {"decision": "deny", "reason": "Only the current conversation's exact diagnostic PNG may be loaded."}
    binding = config.get("image_load_binding")
    if not isinstance(binding, dict):
        return denied, None
    call = payload["toolCall"]
    # The tool name and argument key must come from a real captured hook event.
    tool_name, path_key = binding.get("tool_name"), binding.get("path_argument")
    if tool_name not in ("view_file", "read_file") or not isinstance(path_key, str) or not path_key:
        return denied, None
    if call.get("name") != tool_name:
        return denied, None
    arguments = hook_arguments(call.get("args"))
    if arguments is None or set(arguments) != {path_key}:
        return denied, None
    value = arguments[path_key]
    conversation = payload.get("conversationId")
    expected_sha = binding.get("image_sha256")
    if (not isinstance(value, str) or not isinstance(conversation, str)
            or not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)):
        return denied, None
    try:
        if str(uuid.UUID(conversation)) != conversation:
            return denied, None
        # Reject lexical traversal before resolving or opening anything.
        if any(part in (".", "..") for part in value.replace("\\", "/").split("/")):
            return denied, None
        path, root = Path(value), Path(binding["brain_root"])
        if not path.is_absolute() or not root.is_absolute():
            return denied, None
        relative = path.relative_to(root)
        if (len(relative.parts) != 5 or relative.parts[0] != conversation
                or relative.parts[1:3] != (".system_generated", "steps")
                or not re.fullmatch(r"[0-9]+", relative.parts[3])
                or not re.fullmatch(r"media_[0-9]+\.png", relative.parts[4])):
            return denied, None
        # No final-file or parent-directory symlink is accepted.
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            return denied, None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 8 <= info.st_size <= 4 * 1024 * 1024:
                return denied, None
            with os.fdopen(fd, "rb", closefd=False) as handle:
                data = handle.read(4 * 1024 * 1024 + 1)
            if len(data) != info.st_size or data[:8] != b"\x89PNG\r\n\x1a\n" or sha256(data).hexdigest() != expected_sha:
                return denied, None
        finally:
            os.close(fd)
        evidence = {"kind": "view_load_allowed", "tool": tool_name, "path": str(path),
                    "conversationId": conversation, "image_sha256": expected_sha, "bytes": len(data),
                    "guard_read_exact_png": True, "cli_image_consumption_verified": False}
        return {"decision": "allow", "reason": "Current conversation PNG path and exact diagnostic image SHA verified."}, evidence
    except (OSError, ValueError, KeyError, TypeError):
        return denied, None


def evaluate(payload: Any, config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Do not infer scope from a suffix, substring, or an untrusted argument."""
    result = {"decision": "deny", "reason": "Outside the CPU qualification MCP allowlist."}
    if not isinstance(payload, dict) or not isinstance(payload.get("toolCall"), dict):
        return result, None
    call = payload["toolCall"]
    name = call.get("name")
    if name in ("view_file", "read_file"):
        return inspect_image_load(payload, config)
    bindings = config.get("exact_name_bindings", {})
    tool = bindings.get(name) if isinstance(name, str) else None
    if tool not in ("observe_test", "record_test"):
        return result, None
    # Validate that the host binding itself cannot name another server.
    server = config["server_name"]
    expected = (f"mcp_{server}_{tool}", f"mcp__{server}__{tool}")
    arguments = hook_arguments(call.get("args"))
    if name not in expected or not valid_arguments(tool, arguments):
        return result, None
    return {"decision": "allow", "reason": "Exact scoped CPU qualification MCP tool and schema."}, None


def decide(payload: Any, config: dict[str, Any]) -> dict[str, str]:
    return evaluate(payload, config)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        raw = sys.stdin.buffer.read(131073)
        if len(raw) > 131072:
            raise ValueError("Hook input exceeds the diagnostic bound")
        payload = json.loads(raw)
        decision, image_evidence = evaluate(payload, config)
        append_jsonl(Path(config["audit_path"]), {
            "utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "kind": "PreToolUse", "toolCall": payload.get("toolCall"),
            "conversationId": payload.get("conversationId"), "stepIdx": payload.get("stepIdx"),
            "decision": decision["decision"], "reason": decision["reason"],
            "view_load": image_evidence,
        })
    except Exception as exc:
        # Logging failure must never make an unknown tool executable.
        decision = {"decision": "deny", "reason": f"Qualification guard failed closed ({type(exc).__name__})."}
    print(json.dumps(decision, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
