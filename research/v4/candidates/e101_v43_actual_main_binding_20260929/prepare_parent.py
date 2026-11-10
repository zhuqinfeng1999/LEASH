"""One original generator capture; new runtime selection is forbidden here."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent


def main():
    started = time.perf_counter()
    result = {"kind": "original_E101_source_generator_capture_not_new_selection",
              "pid": os.getpid(), "pid_semantics": "namespace source PID; no host/GPU claim",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "project_main_called": False, "preflight_called": False,
              "runtime_new_selected_called": False, "checker_called": False,
              "Scene": 0, "Consumer": 0, "controls": 0, "GPU": 0, "models": 0}
    try:
        from .capture import capture_parent
        module, source = capture_parent(HERE / "original_capture")
        result.update(passed=True, receipt=module.capture_receipt)
    except BaseException as exc:
        result.update(passed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        result["elapsed_s"] = time.perf_counter() - started
        (HERE / "parent_capture_report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
