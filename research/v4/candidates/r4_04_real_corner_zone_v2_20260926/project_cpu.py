"""Offline full visual-geometry bounds/rays; never establishes native acceptance."""
from itertools import product
import json,time
import mujoco,numpy as np
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix,get_camera_intrinsic_matrix
from .scene import T1R4AdultPillowScene,HERE

def screen(sim):
    m,d=sim.model._model,sim.data._data;groups=np.ones(6,np.uint8);groups[0]=0;entities={}
    for entity,prefix in [('pillow','adult_pillow_1_'),('crib','crib_1_')]:
        rows=[]
        for gid in range(m.ngeom):
            name=m.geom(gid).name or ''
            if name.startswith(prefix) and m.geom_group[gid]==1:
                if int(m.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_MESH):
                    mid=int(m.geom_dataid[gid]);start=int(m.mesh_vertadr[mid]);count=int(m.mesh_vertnum[mid]);local=np.asarray(m.mesh_vert[start:start+count])
                else:
                    bound=m.geom_aabb[gid];local=np.asarray([bound[:3]+bound[3:]*sign for sign in product((-1,1),repeat=3)])
                rows.append((gid,name,d.geom_xpos[gid]+local@d.geom_xmat[gid].reshape(3,3).T,d.geom_xpos[gid].copy()))
        if not rows:raise RuntimeError('missing visual geometry')
        entities[entity]=rows
    views={}
    for camera in ('agentview','robot0_eye_in_hand'):
        k=get_camera_intrinsic_matrix(sim,camera,480,640);extr=get_camera_extrinsic_matrix(sim,camera);inv=np.linalg.inv(extr);origin=extr[:3,3];objects={}
        for entity,geoms in entities.items():
            rows=[];gids={g[0] for g in geoms}
            for gid,name,points,center in geoms:
                c=(inv@np.c_[points,np.ones(len(points))].T)[:3];px=k@c;px=px[:2]/px[2];lo=px.min(1);hi=px.max(1)
                ray=center-origin;ray/=np.linalg.norm(ray);hit=np.array([-1],np.int32);mujoco.mj_ray(m,d,origin,ray,groups,1,-1,hit)
                rows.append({'geom':name,'bbox_xyxy':[*lo.tolist(),*hi.tolist()],
                    'full_geometry_in_frame_12px':bool(np.all(c[2]>0) and lo[0]>12 and lo[1]>12 and hi[0]<628 and hi[1]<468),
                    'center_ray_visible':int(hit[0]) in gids,'center_ray_first_geom':m.geom(int(hit[0])).name if hit[0]>=0 else None})
            objects[entity]={'geoms':rows,'full_geometry_in_frame_12px':all(r['full_geometry_in_frame_12px'] for r in rows),
                'visible_center_fraction':sum(r['center_ray_visible'] for r in rows)/len(rows)}
            # Entity-first rays may hit a wall of the same basket. Report
            # actual material-first rays separately; they are not equivalent.
            token='_pillow_body_' if entity=='pillow' else '_doll_head' 
            material=[r for r in rows if token in r['geom']]
            visible=sum(token in (r['center_ray_first_geom'] or '') for r in material)
            objects[entity]['critical_material_ray_check']={'token':token,'count':len(material),'visible_first_hits':visible,
                'visible_first_hit_fraction':visible/len(material) if material else 0.,'native_visibility_required':True}
        views[camera]=objects
    contacts=[]
    for c in d.contact[:d.ncon]:
        names=[m.geom(int(g)).name or '' for g in (c.geom1,c.geom2)]
        if c.dist<=0 and any(n.startswith(('robot0_','gripper0_')) for n in names):contacts.append({'geoms':names,'distance_m':float(c.dist)})
    return {'views':views,'robot_contacts':contacts,'screen_pass':not contacts and all(o['full_geometry_in_frame_12px'] and o['visible_center_fraction']>.2 for v in views.values() for o in v.values())}

def main():
    out=HERE/'projection_mesh_geometry_v2.json'
    if out.exists():raise ValueError('fresh evidence required')
    start=time.monotonic();rows=[]
    for seed in range(10):
        scene=T1R4AdultPillowScene(seed)
        try:r=screen(scene.backend.sim);r['seed']=seed;rows.append(r)
        finally:scene.close()
    report={'kind':'CPU_FULL_GEOMETRY_SCREEN_NOT_NATIVE_ACCEPTANCE','rows':rows,'passed':all(r['screen_pass'] for r in rows),
        'actual_native_images_required':True,'elapsed_s':time.monotonic()-start}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'elapsed_s':report['elapsed_s']}))
if __name__=='__main__':main()
