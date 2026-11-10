"""Create a disposable Linux worktree with paths bound to this installation."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "LIBERO": ("RLinf/LIBERO", "a8323074d93a09e32bd898630a70531b1f51bc77"),
    "RPent-reproduce-libero": ("RLinf/RPent", "cfd6a3ad8be4d155b6b5a6a0fc671ab0d3350ab7"),
    "RPent-hardware": ("RLinf/RPent", "a8c53119be51e07354b3f97821680b717f88c215"),
}
TEXT = {".py", ".xml", ".json", ".txt", ".obj", ".mhskel", ".mhw", ".mhmat", ".mhclo", ".bddl"}

def git(*args, cwd):
    return subprocess.check_output(["git", *map(str, args)], cwd=cwd, text=True).strip()

def bind_sources(work, cache, physical):
    (work / "upstream").mkdir(exist_ok=True)
    for name, (slug, rev) in REVISIONS.items():
        if name == "RPent-hardware" and not physical:
            continue
        target = work / "upstream" / name
        if cache:
            original = (cache / name).resolve(strict=True)
            if git("rev-parse", "HEAD", cwd=original) != rev:
                raise ValueError(f"{original} does not match revision {rev}")
            target.symlink_to(original, target_is_directory=True)
        else:
            git("clone", "--no-checkout", f"https://github.com/{slug}.git", target, cwd=work)
            git("checkout", "--detach", rev, cwd=target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path, help="New directory outside this checkout")
    parser.add_argument("--upstream-cache", type=Path, help="Use existing pinned upstream checkouts")
    parser.add_argument("--physical-primitives", action="store_true", help="Also prepare the pinned RPent Franka primitive source")
    args = parser.parse_args()
    work = args.worktree.resolve()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 11):
        parser.error("Use the Linux Python 3.11 simulation environment.")
    if work.exists() or work.is_relative_to(ROOT) or not re.fullmatch(r"[/a-zA-Z0-9_.-]+", str(work)):
        parser.error("Choose a new sibling directory with letters, numbers, underscores, hyphens or periods.")
    git("worktree", "add", "--detach", work, "HEAD", cwd=ROOT)
    bind_sources(work, args.upstream_cache, args.physical_primitives)
    marker = b"__LEASH_ROOT__"
    for directory in ("research", "data"):
        for asset in (work / directory).rglob("*"):
            if asset.is_file() and asset.suffix in TEXT:
                data = asset.read_bytes()
                if marker in data:
                    asset.write_bytes(data.replace(marker, os.fsencode(work)))
    cfg = work / "configs/runtime/libero"
    cfg.mkdir(parents=True, exist_ok=True)
    libero = work / "upstream/LIBERO/libero/libero"
    paths = {"assets": work / "data/libero/assets", "bddl_files": libero / "bddl_files",
             "benchmark_root": libero, "datasets": work / "data/libero/datasets", "init_states": libero / "init_files"}
    (cfg / "config.yaml").write_text("".join(f"{key}: {path}\n" for key, path in paths.items()))
    env = work / "envs"
    env.mkdir(exist_ok=True)
    (env / "libero").symlink_to(Path(sys.prefix), target_is_directory=True)
    print(f"cd {work}\nsource tools/activate.sh")

if __name__ == "__main__":
    main()
