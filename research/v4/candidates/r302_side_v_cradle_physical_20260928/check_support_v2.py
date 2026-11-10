"""Recovery of an instrumentation-only contact-index error before any controls."""
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .scene import SideCanTopdownScene

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def contact_rows(model, data, item):
    rows = []
    for cid, c in enumerate(data.contact[:data.ncon]):
        names = [model.geom(int(g)).name or "" for g in (c.geom1, c.geom2)]
        if not (any(item in n for n in names) or any(n.startswith("r302_v_") for n in names)):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, cid, force)
        rows.append({"geoms": names, "dist_m": float(c.dist),
                     "position_m": np.asarray(c.pos).tolist(),
                     "normal_axis_world": np.asarray(c.frame[:3]).tolist(),
                     "normal_force_magnitude_n": float(force[0])})
    return rows


def sample(scene):
    sim = scene.backend.sim
    m, d = sim.model._model, sim.data._data
    obj = scene.backend.objects_dict[scene.item_name]
    j = m.joint(obj.joints[0]).id
    adr, dof = m.jnt_qposadr[j], m.jnt_dofadr[j]
    b = m.body(obj.root_body).id
    rows = contact_rows(m, d, scene.item_name)
    rails = {sign: [c for c in rows if any(f"r302_v_rail_{sign}" == n for n in c["geoms"])
                    and any(scene.item_name in n for n in c["geoms"])]
             for sign in (-1, 1)}
    return {"time_s": float(d.time), "control_sequence": scene.sequence,
            "can_qpos": np.asarray(d.qpos[adr:adr + 7]).tolist(),
            "can_qvel": np.asarray(d.qvel[dof:dof + 6]).tolist(),
            "can_com_world": np.asarray(d.xipos[b]).tolist(),
            "can_axis_world": np.asarray(d.xmat[b]).reshape(3, 3)[:, 2].tolist(),
            "can_vertical_constraint_force_n": float(d.qfrc_constraint[dof + 2]),
            "rails": {str(k): {"contacts": len(v),
                               "normal_force_sum_n": sum(x["normal_force_magnitude_n"] for x in v)}
                      for k, v in rails.items()},
            "contacts": rows}


def main():
    out = HERE / "support_seed0_v2"
    if out.exists():
        raise RuntimeError("First support witness already exists; preserve it")
    out.mkdir()
    scene = None
    try:
        scene = SideCanTopdownScene(0, camera_observations=False)
        model = scene.backend.sim.model._model
        initial = sample(scene)
        samples = [initial]
        for _ in range(20):
            scene.step(np.zeros(7))
            samples.append(sample(scene))
        final = samples[-1]
        rail_ok = all(final["rails"][str(k)]["contacts"] >= 1 and
                      final["rails"][str(k)]["normal_force_sum_n"] > .05
                      for k in (-1, 1))
        qvel = np.asarray(final["can_qvel"])
        source = np.asarray(scene.source_initial_qpos)
        q = np.asarray(final["can_qpos"])
        mg = .4 * abs(float(model.opt.gravity[2]))
        force = final["can_vertical_constraint_force_n"]
        center_drift = float(np.linalg.norm(q[:2] - source[:2]))
        supported = bool(rail_ok and .75 * mg <= force <= 1.25 * mg and
                         np.linalg.norm(qvel[:3]) < .02 and
                         np.linalg.norm(qvel[3:]) < .1 and
                         center_drift < .005 and
                         abs(float(final["can_axis_world"][2])) < .05 and
                         all(not scene.grasp_assist.held for _ in (0,)))
        report = {"kind": "one seed0 source-only physical support witness",
                  "passed": supported,
                  "task_controls": 20, "scripted_grasp_or_transport": False,
                  "initial": initial, "final": final,
                  "all_samples": samples,
                  "rail_force_both": rail_ok,
                  "gravity_load_n": mg,
                  "source_xy_drift_m": center_drift,
                  "held": bool(scene.grasp_assist.held),
                  "source_initial_qpos": source.tolist(),
                  "source_hashes": {
                      str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [HERE / "scene.py", HERE / "check_support_v2.py"]},
                  "meaning": "No task/method success; true dynamic two-rail support only"}
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        mujoco.mj_saveModel(model, str(out / "compiled_model.mjb"))
        d = scene.backend.sim.data._data
        np.savez_compressed(out / "constructor_data.npz", qpos=d.qpos, qvel=d.qvel,
                            act=d.act, ctrl=d.ctrl, time=np.asarray(d.time))
        print(json.dumps({"passed": supported, "final_rails": final["rails"],
                          "constraint_force_n": force,
                          "source_xy_drift_m": center_drift}))
    except BaseException as exc:
        (out / "error.json").write_text(json.dumps({"type": type(exc).__name__,
                                                     "message": str(exc)}) + "\n")
        raise
    finally:
        if scene is not None:
            scene.close()


if __name__ == "__main__":
    main()
