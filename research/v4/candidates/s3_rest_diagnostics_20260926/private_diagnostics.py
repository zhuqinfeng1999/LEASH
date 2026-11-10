"""Read-only, offline-only physical rest evidence; never an online payload.

These diagnostics add no support/velocity/pose requirement to the frozen
scorer. In particular angular velocity and contact load are diagnostic only.
"""
from dataclasses import asdict

import mujoco
import numpy as np

from research.v4.candidates.s3_v42_tabletop_20260926.offline_scorer import S3ControlScorer

REVISION = 's3-offline-rest-components-v1'


def physical_footprint(scene, name, support):
    """Actual collision box/cylinder projections in the mat's local frame.

    For the plate these primitives give an exact projected AABB (including
    current tilt/yaw), not an assumed upright disc. Mesh projections are left
    unknown rather than being invented from a bounding box centre.
    """
    sim=scene.sim
    obj=scene.backend.objects_dict[name]
    sup=scene.backend.objects_dict[support]
    supports=[]
    for geom_name in sup.contact_geoms:
        gid=sim.model.geom_name2id(geom_name)
        if sim.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            supports.append((geom_name,gid))
    if name != 's3_plate_1' or len(supports) != 1:
        return {'exact_primitive_projection_available':False}
    support_name,sid=supports[0]
    rotation=np.asarray(sim.data.geom_xmat[sid]).reshape(3,3)
    center=np.asarray(sim.data.geom_xpos[sid])
    half=np.asarray(sim.model.geom_size[sid])
    geoms=[]
    for geom_name in obj.contact_geoms:
        gid=sim.model.geom_name2id(geom_name)
        local_center=rotation.T@(np.asarray(sim.data.geom_xpos[gid])-center)
        relative=rotation.T@np.asarray(sim.data.geom_xmat[gid]).reshape(3,3)
        size=np.asarray(sim.model.geom_size[gid])
        kind=sim.model.geom_type[gid]
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis=relative[:,2]
            extent=size[0]*np.sqrt(np.maximum(0,1-axis**2))+size[1]*np.abs(axis)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent=np.abs(relative)@size
        else:
            return {'exact_primitive_projection_available':False,'unknown_geom':geom_name}
        geoms.append({'geom':geom_name,'center_support_local_m':local_center.tolist(),
                      'projected_aabb_min_support_local_m':(local_center-extent).tolist(),
                      'projected_aabb_max_support_local_m':(local_center+extent).tolist()})
    lo=np.min([g['projected_aabb_min_support_local_m'] for g in geoms],axis=0)
    hi=np.max([g['projected_aabb_max_support_local_m'] for g in geoms],axis=0)
    return {'exact_primitive_projection_available':True,'support_geom':support_name,
            'support_center_world_m':center.tolist(),'support_rotation_local_to_world':rotation.tolist(),
            'support_half_extents_m':half.tolist(),'object_geoms':geoms,
            'all_collision_geoms_projected_inside_support_xy':bool(np.all(lo[:2]>=-half[:2]) and np.all(hi[:2]<=half[:2])),
            'projected_aabb_min_support_local_m':lo.tolist(),'projected_aabb_max_support_local_m':hi.tolist(),
            'projection_not_a_new_scoring_condition':True}


def contacts_for(scene, name, support_name):
    sim = scene.sim
    own = set(scene.backend.objects_dict[name].contact_geoms)
    support = set(scene.backend.objects_dict[support_name].contact_geoms)
    contacts = []
    total_support_force = np.zeros(3)
    for index, c in enumerate(sim.data.contact[:sim.data.ncon]):
        names = [sim.model.geom_id2name(int(c.geom1)), sim.model.geom_id2name(int(c.geom2))]
        if not any(x in own for x in names):
            continue
        which = 0 if names[0] in own else 1
        wrench = np.zeros(6)
        mujoco.mj_contactForce(sim.model._model, sim.data._data, index, wrench)
        # mjContact.frame normal points geom1 -> geom2 (installed mjdata.h).
        # Contact force is positive along that normal on geom2; the reaction
        # on geom1 has the opposite sign. Record actual solver force, no guess.
        rotation = np.asarray(c.frame).reshape(3, 3)
        force_on_object = rotation.T @ wrench[:3] * (1 if which == 1 else -1)
        named = names[1-which] in support
        if named:
            total_support_force += force_on_object
        contacts.append({
            'contact_index': index, 'geom1': names[0], 'geom2': names[1],
            'object_geom_order': which+1, 'named_support': bool(named),
            'distance_m': float(c.dist), 'qualifies_frozen_support_distance': bool(c.dist <= .002),
            'position_world_m': np.asarray(c.pos).tolist(),
            'contact_frame_world_rows': rotation.tolist(),
            'wrench_contact_force_torque': wrench.tolist(),
            'force_on_object_world_n': force_on_object.tolist(),
        })
    return {
        'pairs': contacts,
        'named_support_force_world_n': total_support_force.tolist(),
        'named_support_vertical_load_n': float(total_support_force[2]),
        'named_support_contact': any(x['named_support'] and x['qualifies_frozen_support_distance'] for x in contacts),
        'load_is_diagnostic_only': True,
    }


def rest_components(scene, name, support, tracker, *, target):
    sim = scene.sim
    obj = scene.backend.objects_dict[name]
    support_obj = scene.backend.objects_dict[support]
    qpos = np.asarray(sim.data.get_joint_qpos(obj.joints[0]), float)
    support_qpos = np.asarray(sim.data.get_joint_qpos(support_obj.joints[0]), float)
    qvel = np.asarray(sim.data.get_joint_qvel(obj.joints[0]), float)
    bid = sim.model.body_name2id(obj.root_body)
    body_velocity_world = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model._model, sim.data._data, mujoco.mjtObj.mjOBJ_BODY,
                            bid, body_velocity_world, 0)
    body_com = np.asarray(sim.data.xipos[bid], float)
    eq = sim.model._model.equality('v42_'+name+'_assist_weld').id
    attached = bool(sim.data.eq_active[eq])
    contact = contacts_for(scene, name, support)
    evidence = asdict(tracker.evidence())
    bits = {'detached': not attached, 'prior_target_detach': evidence['detached_in_target'],
            'in_target': bool(target['in_target']), 'named_support': contact['named_support_contact'],
            'linear_speed_within_limit': bool(np.linalg.norm(qvel[:3]) <= .01)}
    return {
        'object': name, 'support': support,
        'object_freejoint_xyz_world_m': qpos[:3].tolist(), 'object_freejoint_quaternion_wxyz': qpos[3:].tolist(),
        'support_freejoint_xyz_world_m': support_qpos[:3].tolist(),
        'object_com_world_m': body_com.tolist(),
        'body_frame_center_world_m': np.asarray(sim.data.xpos[bid], float).tolist(),
        'object_freejoint_qvel_raw': qvel.tolist(),
        'frozen_linear_velocity_m_s': qvel[:3].tolist(),
        'frozen_linear_speed_m_s': float(np.linalg.norm(qvel[:3])),
        'body_angular_velocity_world_rad_s': body_velocity_world[:3].tolist(),
        'body_linear_velocity_world_m_s': body_velocity_world[3:].tolist(),
        'body_angular_speed_rad_s': float(np.linalg.norm(body_velocity_world[:3])),
        'xy_object_support_center_distance_m': float(np.linalg.norm(qpos[:2]-support_qpos[:2])),
        'target': target, 'contact': contact, 'rest_requirements': bits,
        'support_collision_projection': physical_footprint(scene,name,support),
        'rest_requirements_all_now': bool(all(bits.values())),
        'rest_failures_now': [k for k, value in bits.items() if not value],
        'tracker_evidence': evidence,
        'angular_speed_and_load_are_not_scoring_conditions': True,
    }


def snapshot(scene, scorer):
    sim = scene.sim
    objects = scene.backend.objects_dict
    xyz = lambda name: np.asarray(sim.data.get_joint_qpos(objects[name].joints[0])[:3], float)
    plate = xyz('s3_plate_1'); mat = xyz('s3_placemat_1')
    plate_target = {'kind': 'world_xy_circle', 'center_world_m': mat.tolist(), 'radius_m': .044,
                    'in_target': bool(np.linalg.norm(plate[:2]-mat[:2]) <= .044)}
    sid = sim.model.site_name2id('s3_basket_1_contain_region')
    center = np.asarray(sim.data.site_xpos[sid])
    rotation = np.asarray(sim.data.site_xmat[sid]).reshape(3, 3)
    extent = np.asarray(sim.model.site_size[sid])
    local = rotation.T @ (xyz('s3_ketchup_1')-center)
    ketchup_target = {'kind': 'contain_site_oriented_box', 'center_world_m': center.tolist(),
                      'rotation_local_to_world': rotation.tolist(), 'half_extents_m': extent.tolist(),
                      'object_position_site_local_m': local.tolist(),
                      'in_target': bool(np.all(np.abs(local) <= extent))}
    return {'revision': REVISION, 'offline_only': True,
            'plate': rest_components(scene, 's3_plate_1', 's3_placemat_1', scorer.plate_tracker, target=plate_target),
            'ketchup': rest_components(scene, 's3_ketchup_1', 's3_basket_1', scorer.ketchup_tracker, target=ketchup_target)}


class DiagnosticS3ControlScorer(S3ControlScorer):
    def observe(self, scene):
        # Exactly one call to the original scorer; diagnostics do not step,
        # forward, alter a tracker, or change a returned finish/label.
        state = super().observe(scene)
        state['private_rest_diagnostics'] = snapshot(scene, self)
        return state
