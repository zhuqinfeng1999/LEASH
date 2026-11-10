"""Offline full visual-geometry bounds/rays; never establishes native acceptance."""
from itertools import product
import json,time
import mujoco,numpy as np
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix,get_camera_intrinsic_matrix
from .scene import T1R4MagneticBoxScene,HERE

def screen(sim):
    m,d=sim.model._model,sim.data._data;groups=np.ones(6,np.uint8);groups[0]=0;entities={}
    for entity,prefix in [('beads','magnetic_beads_box_1_'),('crib','crib_1_'),('identity','magnetic_bead_identity_1_')]:
        rows=[]
        for gid in range(m.ngeom):
            name=m.geom(gid).name or ''
            if name.startswith(prefix) and m.geom_group[gid]==1:
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
            token='_bead_' if entity=='beads' else ('_doll_head' if entity=='crib' else '_card_front_visual') 
            material=[r for r in rows if token in r['geom']]
            visible=sum(token in (r['center_ray_first_geom'] or '') for r in material)
            objects[entity]['critical_material_ray_check']={'token':token,'count':len(material),'visible_first_hits':visible,
                'visible_first_hit_fraction':visible/len(material) if material else 0.,'native_visibility_required':True}
        views[camera]=objects
    textviews={}
    layout=json.loads((HERE/'text_layout.json').read_text());card=m.body('magnetic_bead_identity_1_main').id
    for camera in ('agentview','robot0_eye_in_hand'):
        k=get_camera_intrinsic_matrix(sim,camera,480,640);extr=get_camera_extrinsic_matrix(sim,camera);inv=np.linalg.inv(extr);R=d.xmat[card].reshape(3,3);position=d.xpos[card];theta=layout['panel_tilt_y_rad'];rot=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]);pivot=np.asarray(layout['panel_pivot_local_m']);positive=bool(np.dot(extr[:3,3]-position,R@rot[:,0])>0);sign=1 if positive else -1;glyphs=[]
        for glyph in layout['glyphs']:
            x0,y0,x1,y1=glyph['pixel_bbox_xyxy'];local=np.array([[sign*.0016,sign*(u/1360-.5)*.170,.075+(.5-v/840)*.125]for u,v in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]]);local=(local-np.array([0,0,.075]))@rot.T+pivot;world=position+local@R.T;c=(inv@np.c_[world,np.ones(4)].T)[:3];px=k@c;px=(px[:2]/px[2]).T;height=min(np.linalg.norm(px[3]-px[0]),np.linalg.norm(px[2]-px[1]));point=world.mean(0);ray=point-extr[:3,3];ray/=np.linalg.norm(ray);hit=np.array([-1],np.int32);mujoco.mj_ray(m,d,extr[:3,3],ray,groups,1,-1,hit);name=m.geom(int(hit[0])).name if hit[0]>=0 else '';visible='_card_'in(name or'')and'_visual'in(name or'')
            glyphs.append({'text':glyph['text'],'quad_xy_px':px.tolist(),'height_px':float(height),'height_ge12px':bool(height>=12),'whole_glyph_in_frame':bool(np.all(c[2]>0)and np.all(px[:,0]>12)and np.all(px[:,0]<628)and np.all(px[:,1]>12)and np.all(px[:,1]<468)),'actual_first_hit':name,'card_face_first_hit':bool(visible),'native_readability_required':True})
        textviews[camera]=glyphs
    contacts=[]
    for c in d.contact[:d.ncon]:
        names=[m.geom(int(g)).name or '' for g in (c.geom1,c.geom2)]
        if c.dist<=0 and any(n.startswith(('robot0_','gripper0_')) for n in names):contacts.append({'geoms':names,'distance_m':float(c.dist)})
    return {'views':views,'text_pixels':textviews,'text_screen_pass':all(g['height_ge12px']and g['whole_glyph_in_frame']and g['card_face_first_hit']for v in textviews.values()for g in v),'robot_contacts':contacts,'screen_pass':not contacts and all(o['full_geometry_in_frame_12px'] and o['visible_center_fraction']>.2 for v in views.values() for o in v.values())}

def main():
    out=HERE/'projection_full_geometry_v1.json'
    if out.exists():raise ValueError('fresh evidence required')
    start=time.monotonic();rows=[]
    for seed in range(10):
        scene=T1R4MagneticBoxScene(seed)
        try:r=screen(scene.backend.sim);r['seed']=seed;rows.append(r)
        finally:scene.close()
    report={'kind':'CPU_FULL_GEOMETRY_SCREEN_NOT_NATIVE_ACCEPTANCE','rows':rows,'passed':all(r['screen_pass']and r['text_screen_pass']for r in rows),
        'actual_native_images_required':True,'elapsed_s':time.monotonic()-start}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'elapsed_s':report['elapsed_s']}))
if __name__=='__main__':main()
