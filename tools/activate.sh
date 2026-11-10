# Source this in bash after activating the Python environment.
LEASH_WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$LEASH_WORKTREE:$LEASH_WORKTREE/upstream/RPent-reproduce-libero:$LEASH_WORKTREE/upstream/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$LEASH_WORKTREE/configs/runtime/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
