"""Match RPent's saved low-resolution calibration to its actual public depth.

The original serializer assumes a 256-square agentview. Project observations
are 640 by 480. Keep its image/depth conventions and high-resolution path,
and adapt only that metadata request without editing the reproduction.
"""
from types import SimpleNamespace

import numpy as np


def save_public_artifacts(original_save, primitives, state, step, raw):
    image = np.asarray(raw["agentview_image"])
    depth = np.asarray(raw["agentview_depth"])
    if image.ndim != 3 or image.shape[-1] != 3 or depth.shape[:2] != image.shape[:2]:
        raise ValueError("agentview RGB and depth dimensions must agree")
    height, width = map(int, depth.shape[:2])
    env = primitives.env

    def camera_meta(camera_name="agentview", height=256, width=256):
        if camera_name == "agentview" and (height, width) == (256, 256):
            height, width = image.shape[:2]
        return env.get_camera_meta(camera_name, int(height), int(width))

    def save(name, value, **kwargs):
        if name == "agentview_metadata.json":
            value = dict(value)
            value["projection"] = (
                "Use back_project(row, col, step=NN) with its high-resolution "
                "default for agentview_high.png. Use resolution='low' only for "
                f"the matching {width}x{height} agentview.png calibration frame."
            )
        return state.save(name, value, **kwargs)

    # These are precisely the serializer's two inputs; no simulator truth or
    # extra render, sampling, or world-map readback is introduced.
    cameras = SimpleNamespace(get_camera_meta=camera_meta, render_camera=env.render_camera)
    public = SimpleNamespace(_last_obs=primitives._last_obs, env=cameras)
    return original_save(public, SimpleNamespace(save=save), step, raw)


def install_public_artifact_dimensions():
    """Install once in this project's worker, before RPent's initial dump."""
    from robots.libero import tools

    original = tools._save_observation_artifacts
    if getattr(original, "_safevla_public_dimensions", False):
        return

    def save(primitives, state, step, raw):
        return save_public_artifacts(original, primitives, state, step, raw)

    save._safevla_public_dimensions = True
    tools._save_observation_artifacts = save



def install_initial_occupancy_state():
    """Extend the current serializer's robot state without replacing its format."""
    from robots.libero import tools
    original_dump = tools.dump_state
    if getattr(original_dump, '_safevla_initial_occupancy', False):
        return
    def dump(primitives, state, log=None):
        scene = getattr(primitives.env, 'scene', None)
        fact = getattr(scene, 'public_initial_occupancy', lambda: None)()
        observations = getattr(scene, 'public_grasp_observations', lambda: None)()
        if observations is not None:
            packet = scene.capture_agent()  # same frozen primitive-return state, no step
            observations.observe(packet)  # fixed actual primitive-return checkpoint
            observations.tool_return(log or {})
        if fact is None and observations is None:
            return original_dump(primitives, state, log)
        class PublicState:
            def __getattr__(self, name):
                return getattr(state, name)
            def record_step(self, **kwargs):
                kwargs['state'] = dict(kwargs['state'])
                if fact is not None:
                    kwargs['state']['initial_occupancy'] = fact
                if observations is not None:
                    kwargs['state']['executed_grasp_observations'] = observations.metadata(include_geometry=False)
                return state.record_step(**kwargs)
        record = original_dump(primitives, PublicState(), log)
        if observations is not None and observations.last is not None:
            observations.last['artifact_step'] = record.step_idx
        return record
    dump._safevla_initial_occupancy = True
    tools.dump_state = dump
