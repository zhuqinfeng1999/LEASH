from pathlib import Path
import json,os,argparse
from contextlib import contextmanager

@contextmanager
def native_scene(task, seed=0, method="Ours", *, camera_observations=False):
    """Open the native RoboRisk-S scene selected by the experiment runner."""
    from research.v4.bridge.run_b0 import lite_runner
    tasks = json.loads((Path(__file__).resolve().parent / "sim-tasks.json").read_text(encoding="utf8"))
    if task not in {row["id"] for row in tasks}:
        raise ValueError(f"Unknown paper task ID: {task}")
    if seed not in (0, 1, 2) or method not in ("B0", "Ours"):
        raise ValueError("Expected one of the three benchmark seeds and B0 or Ours")
    source_id = "R" + task[1:] if task.startswith("T") else task
    os.environ.update(SAFEVLA_LEASH_METHOD=method, SAFEVLA_LEASH_BACKEND="scripted")
    module = lite_runner(source_id)
    if source_id.startswith("R3-"):
        factory = module.OWN_ENTRY.LowerAimR3Scene
    else:
        factory = next(getattr(module, n) for n in ("Scene", "OWN_SCENE", "S1StablePan300Facade") if hasattr(module, n))
    scene = factory(seed=seed, camera_observations=True, render_gpu_device_id=0) if camera_observations else factory(seed=seed)
    try:
        yield scene
    finally:
        scene.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset one native simulated task")
    parser.add_argument("task", help="Paper task ID, e.g. T1-01 or P3-01")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    with native_scene(args.task, args.seed):
        print(f"Reset {args.task} at seed {args.seed}")
