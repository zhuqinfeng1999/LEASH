"""Read-only whitelists when a LEASH callback receives an adapter/toolkit."""
from types import MappingProxyType


class PublicAccess:
    __slots__ = ("_allowed", "_dynamic")

    def __init__(self, target, names, *, dynamic=(), extra=None):
        # Retain only permitted values; never retain an env/sim target handle.
        object.__setattr__(self, "_allowed", MappingProxyType({
            name: getattr(target, name) for name in names if hasattr(target, name)
        } | (extra or {})))
        object.__setattr__(self, "_dynamic", MappingProxyType({
            name: (lambda name=name: getattr(target, name)) for name in dynamic
        }))

    def __getattr__(self, name):
        try:
            dynamic = object.__getattribute__(self, "_dynamic")
            return dynamic[name]() if name in dynamic else object.__getattribute__(self, "_allowed")[name]
        except KeyError:
            raise AttributeError(f"LEASH public whitelist forbids {name!r}") from None

    def __setattr__(self, name, value):
        raise AttributeError("LEASH public whitelist is read-only")


def public_adapter(adapter):
    def review_context(*, instruction=None, history=None):
        if instruction is not None:
            adapter.instruction = instruction
        if history is not None:
            adapter.history = history
    return PublicAccess(adapter, ("estimator", "dispatcher", "invalidate_binding",
                                  "primitive_scope", "_proprio"),
                        dynamic=("instruction", "history"),
                        extra={"set_review_context": review_context})


def public_toolkit(toolkit):
    state = tuple(name for name in ("round_finished", "finish_result", "episode_calls",
                                    "round_calls", "per_round_budget", "episode_budget")
                  if hasattr(toolkit, name))
    extra = {}
    if hasattr(toolkit, "original"):
        # RPent's ordinary observation audit reads the saved public state.
        # The nested whitelist never exposes its environment or primitives.
        extra["original"] = PublicAccess(toolkit.original, ("state",))
    return PublicAccess(toolkit, ("execute_tool", "begin_round", "get_tools_spec", "calls",
                                  "cancel_active_and_wait", "close", "memory_dir"),
                        dynamic=state, extra=extra)


def public_scene(scene):
    # Environment access is confined to this public input construction boundary.
    dt = float(scene.env.env.control_timestep)
    return PublicAccess(scene, ("capture_agent", "public_grasp_observations"), dynamic=("sequence", "instruction"),
                        extra={"control_timestep": dt})


def public_primitives(primitives):
    return PublicAccess(primitives, ("release", "set_gripper", "move_to", "move_pose",
                                    "rotate_pitch", "rotate_wrist",
                                    "vla_act", "pi0_pick", "pi0_doubled"))
