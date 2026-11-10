"""Private task reception: main bottle body in actual basket, floor supported.

The neck/cap may project beyond the opening. Existing grasp/lift/OPEN/support
window and all danger predicates are unchanged. No online state/control change.
"""
import numpy as np
REVISION='basket-main-body-floor-reception-dev10-20261014'
TOLERANCE_M=.003

def reception_evidence(geometry):
    if not isinstance(geometry,dict):raise ValueError('Actual basket geometry required')
    lower=np.asarray(geometry['interior_xy_min_m'],float);upper=np.asarray(geometry['interior_xy_max_m'],float)
    origin=np.asarray(geometry['frame_origin_world_m'],float);rotation=np.asarray(geometry['frame_rotation_local_to_world'],float)
    if lower.shape!=(2,)or upper.shape!=(2,)or origin.shape!=(3,)or rotation.shape!=(3,3)or not all(np.isfinite(a).all()for a in(lower,upper,origin,rotation))or np.any(upper<=lower):raise ValueError('Actual receiving frame/bounds are missing')
    bodies=[p for p in geometry['full_bottle_collision_projections']if p.get('geom','').endswith('_bottle_contact_base_body')]
    if len(bodies)!=1:raise ValueError('Actual main bottle body projection is required')
    lo=np.asarray(bodies[0]['min_local_m'],float);hi=np.asarray(bodies[0]['max_local_m'],float)
    if lo.shape!=(3,)or hi.shape!=(3,)or not(np.isfinite(lo).all()and np.isfinite(hi).all())or np.any(hi<lo):raise ValueError('Actual main body bounds are invalid')
    center=(lo[:2]+hi[:2])/2
    main_body_inside=bool(np.all(center>=lower-TOLERANCE_M)and np.all(center<=upper+TOLERANCE_M))
    contacts=[]
    for contact in geometry['floor_contacts']:
        if 'upward_load_n'not in contact or type(contact.get('qualifies_original_distance'))is not bool:raise ValueError('Actual contact load/distance evidence is required')
        if type(contact['upward_load_n'])not in(int,float)or not np.isfinite(contact['upward_load_n']):raise ValueError('Actual floor load must be finite')
        position=np.asarray(contact['position_world_m'],float)
        if position.shape!=(3,)or not np.isfinite(position).all():raise ValueError('Actual basket floor contact point is invalid')
        local=rotation.T@(position-origin)
        if geometry['floor_geom']in(contact.get('geom1'),contact.get('geom2'))and contact.get('qualifies_original_distance')is True and contact.get('upward_load_n',0)>0 and np.all(local[:2]>=lower-TOLERANCE_M)and np.all(local[:2]<=upper+TOLERANCE_M):contacts.append(local.tolist())
    if type(geometry.get('floor_supported'))is not bool or type(geometry.get('bottom_inside_rim'))is not bool:raise ValueError('Actual floor/bottom support flags are required')
    accepted=bool(main_body_inside and contacts and geometry['floor_supported']and geometry['bottom_inside_rim'])
    return dict(revision=REVISION,in_target=accepted,main_body_center_local_xy_m=center.tolist(),main_body_reference_inside=main_body_inside,receiving_floor_load_contact_points_local_m=contacts,actual_floor_supported=geometry['floor_supported'],bottom_inside_rim=geometry['bottom_inside_rim'],neck_cap_overhang_permitted=True,tolerance_m=TOLERANCE_M,full_bottle_inside_diagnostic=geometry.get('entire_bottle_footprint_inside'))

def body_supported_basket_target(geometry):return reception_evidence(geometry)['in_target']
