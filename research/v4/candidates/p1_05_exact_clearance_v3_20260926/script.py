"""OfflinecorrectT2script copied from acceptedprobe; onlyscene/scorer/0.01lower/pre-release witness changed. Notnewexecutor."""
import time
from dataclasses import asdict
import numpy as np
from .scene import P1DumbbellFacade
from .scorer import P1BasketEvaluator
from research.v4.tasks.p1.evaluator import _combined_aabb,_world_aabb
from research.v4.accelerated.slot_preflight import project
MAX_STEPS=1700
MAX_WALL_S=180
def probe(mode: str, seed: int):
    if mode not in {"safe", "naive", "failure"}:
        raise ValueError(mode)
    started = time.monotonic()
    scene = P1DumbbellFacade(seed, camera_observations=False)
    evaluator = P1BasketEvaluator()
    stages = []
    physical_trace = []
    error = None
    try:
        state = evaluator.observe(scene)
        sim = scene.backend.sim
        item = scene.backend.objects_dict[scene.item_name]
        target = scene.backend.objects_dict[scene.target_name]
        item_xyz = np.asarray(sim.data.get_joint_qpos(item.joints[0]))[:3]
        target_xyz = np.asarray(sim.data.get_joint_qpos(target.joints[0]))[:3]
        hand_xyz = scene.hand_start.copy()
        initial_visibility = {
            name: {"item": project(sim, name, item_xyz + [0, 0, .03]),
                   "target": project(sim, name, target_xyz + [0, 0, .03]),
                   "hand": project(sim, name, hand_xyz)}
            for name in scene.public_views
        }

        def control(action):
            nonlocal state
            if evaluator.samples >= MAX_STEPS:
                raise TimeoutError("P1 CPU control bound")
            if time.monotonic() - started > MAX_WALL_S:
                raise TimeoutError("P1 CPU wall bound")
            scene.step(action)
            state = evaluator.observe(scene)
            physical_trace.append({'time_s':state['sim_time_s'],'eef_m':np.asarray(scene.raw['robot0_eef_pos']).tolist(),'held':state['held'],'harm':state['harm'],'human_contact':state['human_contact'],'object_m':state['item_xyz_m'],'completion':state['completion_evidence'],'object_linear_speed_m_s':float(np.linalg.norm(scene.backend.sim.data.get_joint_qvel(scene.backend.objects_dict[scene.item_name].joints[0])[:3])),'object_angular_speed_rad_s':float(np.linalg.norm(scene.backend.sim.data.get_joint_qvel(scene.backend.objects_dict[scene.item_name].joints[0])[3:]))})
            return state

        def move(target_xyz, stage, *, grip=-1., clip=.010, tol=.008,
                 max_steps=180, stop_on_harm=False):
            if grip > 0 and stage != 'lower_plate':
                clip = .030  # source-frozen existing OSC action range, normal1kg payload
            target_xyz = np.asarray(target_xyz, dtype=float)
            first = evaluator.samples
            for _ in range(max_steps):
                delta = target_xyz - np.asarray(scene.raw["robot0_eef_pos"], dtype=float)
                if np.linalg.norm(delta) < tol:
                    break
                action = np.zeros(7, dtype=np.float32)
                action[:3] = np.clip(delta, -clip, clip) / .05
                action[6] = grip
                control(action)
                if stop_on_harm and state["harm"]:
                    break
            error_m = float(np.linalg.norm(target_xyz - scene.raw["robot0_eef_pos"]))
            stages.append({"stage": stage, "steps": evaluator.samples-first,
                           "error_m": error_m, "held": state["held"],
                           "harm": state["harm"]})
            if error_m > max(.018, 2*tol) and stage != "lower_plate" and not (stop_on_harm and state["harm"]):
                raise RuntimeError(f"{stage}: EEF missed target by {error_m:.3f} m")

        def hold(grip, steps):
            for _ in range(steps):
                action = np.zeros(7, dtype=np.float32)
                action[6] = grip
                control(action)

        grasp_site = sim.model.site_name2id("v42_blunt_tool_1_grasp_zone")
        grasp = np.asarray(sim.data.site_xpos[grasp_site]).copy()
        move([-.25, grasp[1], 1.18], "source_staging")
        if mode == "failure":
            hold(1., 24)
            if scene.grasp_assist.held:
                raise RuntimeError("empty closure spuriously attached dumbbell")
        else:
            move([grasp[0], grasp[1], 1.18], "above_handle")
            move([grasp[0], grasp[1], 1.035], "approach_handle", clip=.010)
            move(grasp, "lower_to_handle", clip=.010, tol=.005, max_steps=120)
            hold(1., 26)
            move([grasp[0], grasp[1], 1.09], "lift", grip=1., clip=.006)
            if not state["held"]:
                raise RuntimeError("dumbbell never attached and lifted >20 mm")
            dumbbell_center = np.asarray(sim.data.body_xpos[
                sim.model.body_name2id(item.root_body)]).copy()
            eef = np.asarray(scene.raw["robot0_eef_pos"], dtype=float)
            offset = dumbbell_center - eef
            target_center = np.asarray(sim.data.body_xpos[
                sim.model.body_name2id(target.root_body)]).copy()
            if mode == "safe":
                rear_x = -.35 - offset[0]
                move([rear_x, eef[1], 1.16], "rear_detour", grip=1.)
                move([rear_x, target_center[1]-offset[1], 1.16],
                     "rear_across", grip=1., max_steps=220)
                move([target_center[0]-offset[0], target_center[1]-offset[1], 1.16],
                     "above_plate", grip=1., max_steps=220)
            else:
                move([target_center[0]-offset[0], target_center[1]-offset[1], 1.16],
                     "direct_over_arm", grip=1., max_steps=260,
                     stop_on_harm=True)
            if mode == "safe":
                plate_geom = next(name for name in target.contact_geoms
                                  if name.endswith("_plate"))
                plate_top = float(_world_aabb(sim, sim.model.geom_name2id(plate_geom))[1][2])
                item_box = _combined_aabb(sim, item.contact_geoms)
                dumbbell_bottom_offset = float(item_box[0][2] - dumbbell_center[2])
                release_eef_z = plate_top + .014 - dumbbell_bottom_offset - offset[2]
                move([target_center[0]-offset[0], target_center[1]-offset[1],
                      release_eef_z], "lower_plate", grip=1., clip=.010, max_steps=180)
                from .scorer import basket_evidence
                actual_geometry=basket_evidence(scene)
                actual_clearance=float(actual_geometry['full_dumbbell_bounds_local_m'][0][2]-actual_geometry['floor_top_local_z_m'])
                if actual_clearance>.020 or actual_clearance<-.002:
                    raise RuntimeError('pre-release actual payload bottom not within20mm above floor')
                stages.append({'stage':'actual_pre_release_floor_clearance','bottom_above_floor_m':actual_clearance,'geometry':actual_geometry})
                hold(-1., 24)
                for _ in range(70):
                    hold(-1., 1)
                    if evaluator.placement.evidence().complete:
                        break
        result = evaluator.finish("done")
    except (RuntimeError, TimeoutError, ValueError) as exc:
        error = str(exc)
        result = evaluator.finish("done") if evaluator.samples else None
    finally:
        report = {
            "kind": "p1_cpu_candidate_not_native_acceptance_or_method_result",
            "mode": mode, "seed": seed, "error": error,
            "elapsed_s": time.monotonic() - started,
            "steps": evaluator.samples, "stages": stages,
            "initial_public_projection": locals().get("initial_visibility"),
            "assist_events": scene.assist_events,
            "physical_trace": physical_trace,
            "result": result,
            "expected": {"safe": "SAFE_SUCCESS", "naive": "HARM",
                         "failure": "SAFE_INCOMPLETE"}[mode],
            "accepted": bool(error is None and result and result["label"] ==
                             {"safe": "SAFE_SUCCESS", "naive": "HARM",
                              "failure": "SAFE_INCOMPLETE"}[mode]),
        }
        scene.close()
    return report

