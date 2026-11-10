"""Proposed offline full-shape measurement correction; not approved integration.

Same100mm whole footprint/floor/support criteria. Evaluate actual compiled
collision and visual surfaces, never invented XY-box corner combinations.
"""
import mujoco,numpy as np
from research.v4.candidates.p1_06_wine_bucket_t2_20260926 import scorer as original
from research.v4.candidates.p1_06_exact_shape_audit_20260926.replay import bounds_in_target

def apply_exact_fullshape(original_geometry,shapes):
    if not shapes:raise ValueError('missing actual full-shape evidence')
    row=dict(original_geometry)
    radius=max(float(s['radial_upper_m'])for s in shapes)
    lower=float(row['full_bottle_bounds_local_m'][0][2]);floor=float(row['floor_top_local_z_m'])
    row.update(legacy_aabb_corner_radius_m=float(row['footprint_radius_bound_m']),footprint_radius_bound_m=radius,
               exact_shape_measurement='actual compiled collision+visual vertices/cap circles with conservative discretization bound',
               in_target=bool(radius<=float(row['opening_radius_m'])+1e-6 and -.002<=lower-floor<=.020))
    return row

def exact_shapes(scene):
    sim=scene.backend.sim;m=sim.model._model;d=sim.data._data;item=scene.backend.objects_dict[scene.item_name];target=scene.backend.objects_dict[scene.target_name];tid=sim.model.body_name2id(target.root_body)
    target_pose=dict(target_origin_m=d.body_xpos[tid].tolist(),target_rotation=d.body_xmat[tid].reshape(3,3).tolist())
    shapes=[]
    for gid in range(m.ngeom):
        name=m.geom(gid).name or''
        if not name.startswith(scene.item_name+'_')or int(m.geom_group[gid])not in(0,1):continue
        definition=dict(name=name,type=int(m.geom_type[gid]),size=m.geom_size[gid].tolist(),collision=name in item.contact_geoms)
        if definition['type']==int(mujoco.mjtGeom.mjGEOM_MESH):
            mid=int(m.geom_dataid[gid]);start=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);definition['compiled_vertices']=m.mesh_vert[start:start+n]
        pose=dict(origin_m=d.geom_xpos[gid].tolist(),rotation=d.geom_xmat[gid].reshape(3,3).tolist())
        shape=bounds_in_target(definition,pose,target_pose);shape['inside_unchanged100mm']=bool(shape['inside_unchanged100mm']);shapes.append(shape)
    return shapes

def bucket_evidence(scene):return apply_exact_fullshape(original.bucket_evidence(scene),exact_shapes(scene))
class P1BucketEvaluator(original.P1BucketEvaluator):
    def observe(self,scene):
        self.current_geometry=bucket_evidence(scene)
        row=super(original.P1BucketEvaluator,self).observe(scene)
        row.update(item_on_target=bool(self.current_geometry['in_target']and self.current_geometry['floor_supported']and not row['attached']),case_completion=self.current_geometry)
        return row
