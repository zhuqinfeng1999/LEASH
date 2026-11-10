"""Read-only actual initialization/control evidence; no forward or restore."""
import numpy as np
import mujoco

PAD_NAMES = ('gripper0_right_finger1_pad_collision', 'gripper0_right_finger2_pad_collision')


def contacts(scene):
    m, d = scene.backend.sim.model._model, scene.backend.sim.data._data
    rows = []
    for i, c in enumerate(d.contact[:d.ncon]):
        names = [m.geom(int(g)).name or '' for g in (c.geom1, c.geom2)]
        cold_pad = (any(n in PAD_NAMES for n in names) and
                    any(n == 'moka_pot_1_contact_06' or n.startswith('moka_pot_1_cold_loop_collision_') for n in names))
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        rows.append({'geoms': names, 'dist_m': float(c.dist), 'world_pos': c.pos.tolist(),
                     'contact_frame': c.frame.tolist(), 'contact_force': force.tolist(),
                     'soft_pad_real_cold_material': cold_pad,
                     'hard_stop': bool(c.dist < -.001 and not cold_pad)})
    return rows


def snapshot(scene):
    m, d = scene.backend.sim.model._model, scene.backend.sim.data._data
    robot = scene.backend.robots[0]
    part = robot.composite_controller.part_controllers['right']
    row = {'sequence': int(scene.sequence), 'time_s': float(d.time),
           'qpos': d.qpos.tolist(), 'qvel': d.qvel.tolist(), 'ctrl': d.ctrl.tolist(),
           'eq_active': d.eq_active.tolist(), 'eq_data': m.eq_data.tolist(),
           'eef_pos': np.asarray(scene.raw['robot0_eef_pos']).tolist(),
           'eef_quat': np.asarray(scene.raw['robot0_eef_quat']).tolist(),
           'robot_init_qpos': np.asarray(robot.init_qpos).tolist(),
           'robot_joint_qpos': np.asarray(robot._joint_positions).tolist(),
           'OSC_initial_joint': np.asarray(part.initial_joint).tolist(),
           'base_world': d.xpos[m.body('robot0_base').id].tolist(),
           'item_world': d.xpos[m.body('moka_pot_1_main').id].tolist(),
           'item_mass_kg': float(m.body_mass[m.body('moka_pot_1_main').id]),
           'held': bool(scene.grasp_assist.held), 'assist_events': list(scene.assist_events),
           'contacts': contacts(scene)}
    return row


def assert_initial(scene, *, before_neutral):
    from .scene import INITIAL_JOINTS
    row = snapshot(scene)
    if row['item_mass_kg'] != .420:
        raise ValueError('Own Moka mass changed')
    if row['robot_init_qpos'] != list(INITIAL_JOINTS) or row['OSC_initial_joint'] != list(INITIAL_JOINTS):
        raise ValueError('Normal reset/OSC initial joint references differ')
    if before_neutral and row['robot_joint_qpos'] != list(INITIAL_JOINTS):
        raise ValueError('Normal reset actual initial q differs')
    if row['base_world'] != [-.61, 0., .912]:
        raise ValueError('Own normal base placement differs')
    if any(c['hard_stop'] for c in row['contacts']):
        raise RuntimeError('First actual initialization hard contact >1mm')
    if scene.grasp_assist.held:
        raise RuntimeError('Source attached during OPEN common initialization')
    if not before_neutral and not any(
            'r301_source_wood_riser_solid' in c['geoms'] and
            any(n.startswith('moka_pot_1_contact_') for n in c['geoms'])
            for c in row['contacts']):
        raise RuntimeError('Actual Moka metal-foot support missing during20 OPEN')
