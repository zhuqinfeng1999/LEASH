"""One actual zero-task-motion support witness and public-camera projection."""

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from research.v4.candidates.r302_side_v_cradle_physical_20260928.check_support_v2 import sample
from .scene import RearPresentedCanScene


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def projection(model, data, camera, world_point):
    cid = model.camera(camera).id
    delta = data.cam_xmat[cid].reshape(3, 3).T @ (world_point - data.cam_xpos[cid])
    focal = 480 / (2 * math.tan(math.radians(model.cam_fovy[cid]) / 2))
    uv = [320 + focal * delta[0] / -delta[2], 240 - focal * delta[1] / -delta[2]]
    return {"camera": camera, "uv_640x480": uv, "forward_depth_m": -float(delta[2]),
            "within_frame": bool(delta[2] < 0 and 0 <= uv[0] < 640 and 0 <= uv[1] < 480)}


def main():
    out = HERE / "support_seed0_v2"
    if out.exists():
        raise RuntimeError("One-shot source support evidence already exists")
    out.mkdir()
    scene = None
    try:
        scene = RearPresentedCanScene(0, camera_observations=False)
        sim = scene.backend.sim
        model, data = sim.model._model, sim.data._data
        initial = sample(scene)
        samples = [initial]
        for _ in range(20):
            scene.step(np.zeros(7))
            samples.append(sample(scene))
        final = samples[-1]
        rail_ok = all(final["rails"][str(k)]["contacts"] >= 1 and
                      final["rails"][str(k)]["normal_force_sum_n"] > .05
                      for k in (-1, 1))
        source = np.asarray(scene.source_initial_qpos)
        q = np.asarray(final["can_qpos"])
        vel = np.asarray(final["can_qvel"])
        gravity_load = .4 * abs(float(model.opt.gravity[2]))
        supported = bool(rail_ok and
                         .75 * gravity_load <= final["can_vertical_constraint_force_n"] <= 1.25 * gravity_load and
                         np.linalg.norm(vel[:3]) < .02 and np.linalg.norm(vel[3:]) < .1 and
                         np.linalg.norm(q[:2] - source[:2]) < .005 and
                         abs(final["can_axis_world"][2]) < .05 and not scene.grasp_assist.held)
        site = data.site_xpos[model.site("v42_steel_thermos_1_grasp_zone").id].copy()
        center = data.xpos[model.body("steel_thermos_1_main").id].copy()
        cameras = {name: {"can_center": projection(model, data, name, center),
                          "original_plus42_grasp_site": projection(model, data, name, site)}
                   for name in ("agentview", "robot0_eye_in_hand")}
        report = {"kind": "one seed0 actual neutral support after common-center 180deg yaw",
                  "passed": supported and all(v["can_center"]["within_frame"] and
                                               v["original_plus42_grasp_site"]["within_frame"]
                                               for v in cameras.values()),
                  "support_passed": supported, "task_controls": 20,
                  "scripted_grasp_or_transport": False, "initial": initial,
                  "final": final, "all_samples": samples,
                  "camera_geometry_only_not_native_pixel_readability": cameras,
                  "source_initial_qpos": source.tolist(), "gravity_load_n": gravity_load,
                  "held": bool(scene.grasp_assist.held),
                  "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (HERE / "scene.py", HERE / "support.py")}}
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        mujoco.mj_saveModel(model, str(out / "compiled_model.mjb"))
        np.savez_compressed(out / "constructor_data.npz", qpos=data.qpos,
                            qvel=data.qvel, act=data.act, ctrl=data.ctrl,
                            time=np.asarray(data.time))
        print(json.dumps({"passed": report["passed"], "support": supported,
                          "rails": final["rails"], "camera_geometry": cameras}))
    except BaseException as exc:
        (out / "error.json").write_text(json.dumps({"type": type(exc).__name__,
                                                     "message": str(exc)}) + "\n")
        raise
    finally:
        if scene is not None:
            scene.close()


if __name__ == "__main__":
    main()
