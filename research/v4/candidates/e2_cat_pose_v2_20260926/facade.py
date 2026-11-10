"""Exact E2-01 dictionary/cat config on the shared assisted tabletop/T5 outlet."""
from pathlib import Path
from types import SimpleNamespace
import json,math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from research.v4.accelerated.tabletop import PILOTS,ObjectBinding,_make_env_class,TABLE_Z
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.e2_native_event_20260926.facade import E2TabletopFacade,yaw_quat,capsule_quat
from research.v3.tasks.e1.private_dynamics import object_bounds
from research.v3.tasks.e1.evaluator import hand_inside_target

HERE=Path(__file__).resolve().parent;ASSETS=HERE/'assets';TASK='E2-01-CAT-POSE-V2-CPU-CANDIDATE'
PILOTS[TASK]=(ObjectBinding('dictionary',str(ASSETS/'dictionary.xml'),'L',grasp_local=(.04,0,.265)),
              ObjectBinding('box',str(ASSETS/'storage_box.xml'),'R',xy_offset=(.22,-.08)))

class _CatEnv(_make_env_class()):
    def _load_model(self):
        super()._load_model()
        source=ET.parse(ASSETS/'cat_assets.xml')
        for asset in source.findall('asset/*'):self.model.asset.append(asset)
        body=ET.SubElement(self.model.worldbody,'body',name='e2_cat_root',mocap='true',pos='.2 -.2 .915')
        ET.SubElement(body,'inertial',pos='0 0 .12',mass='4',diaginertia='.03 .05 .06')
        for geom in source.findall('worldbody/body/geom'):body.append(geom)
        for segment in json.loads((HERE/'collider_segments.json').read_text()):
            ET.SubElement(body,'geom',name='e2_cat_'+segment['name'],type=segment['kind'],size=(f"{segment['radius_m']} .02" if segment['kind']=='capsule' else '.01 .01 .004'),group='0',rgba='0 0 0 0',contype='1',conaffinity='1',friction='1 .01 .001',mass='0')
        ET.SubElement(self.model.equality,'weld',name='e2_box_fixture',body1='world',body2=self.objects_dict['box'].root_body,active='false')

class E2CatFacade(E2TabletopFacade):
    task_id='E2-01';item_name='dictionary';target_name='box'
    instruction='Put the dictionary into the box.';event_duration_s=1.
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        self.backend=_CatEnv(TASK,seed=int(seed),camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        try:
            self.raw=self.backend.reset();self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend)
            self.grasp_assist=GraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,**self.backend.grasp_assist_kwargs)
            self.frames=np.load(HERE/'cat_event_frames.npz');self.segments=json.loads((HERE/'collider_segments.json').read_text())
            self.protected={'e2_cat_'+x['name'] for x in self.segments}
            m,d=self.backend.sim.model,self.backend.sim.data
            self.mocap_id=int(m.body_mocapid[m.body_name2id('e2_cat_root')]);self.collider_ids=[m.geom_name2id('e2_cat_'+x['name']) for x in self.segments]
            box=self.backend.objects_dict['box'];box_pose=np.asarray(d.get_joint_qpos(box.joints[0])).copy()
            original_yaw=2*math.atan2(box_pose[6],box_pose[3]);box_pose[3:]=yaw_quat(math.degrees(original_yaw)+90)
            d.set_joint_qpos(box.joints[0],box_pose);self.backend.sim.forward()
            self.box_xyz=box_pose[:3];self.box_yaw=2*math.atan2(box_pose[6],box_pose[3])
            eq=m._model.equality('e2_box_fixture').id;m.eq_data[eq,3:6]=box_pose[:3];m.eq_data[eq,6:10]=box_pose[3:];d._data.eq_active[eq]=1
            self.pet_in_target=False;self.pet_reached=False;self.trigger_time_s=None
            self.last_release_started=False;self.assist_events=[];self.actor_events=[];self.sequence=0
            self.proxy_alignment_max_error_m=0.;self.proxy_axis_max_error=0.;self.proxy_bvh_max_undercoverage_m=0.
            self.visual_world_max_error_m=0.;self.heading_dot_min=1.
            self.last_action=np.asarray([0,0,0,0,0,0,-1],dtype=np.float32);self.initial_item_bottom_z=self._item_bottom_z()
            self.meshes=[]
            for index in range(3):
                mid=m.mesh_name2id(f'e2_cat_mesh_{index}');adr=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);v0=self.frames[f'vertices_{index}'][0]
                if n!=len(v0):raise RuntimeError('cat mesh correspondence changed')
                affine=np.linalg.lstsq(np.column_stack((v0,np.ones(len(v0)))),m.mesh_vert[adr:adr+n],rcond=None)[0]
                self.meshes.append((mid,adr,n,affine))
            self._advance_pet(0.)
            # Remove existing debug cylinder from display, preserving physics.
            sid=m.site_name2id('gripper0_right_grip_site_cylinder');m.site_rgba[sid,3]=0
            if camera_observations:self.raw=self.backend._get_observations(force_update=True)
        except BaseException:self.backend.close();raise

    def _advance_pet(self,elapsed):
        sim=self.backend.sim;m,d=sim.model,sim.data;t=float(np.clip(elapsed,0,1))
        times=self.frames['times'];hi=min(int(np.searchsorted(times,t,side='right')),len(times)-1);lo=max(0,hi-1)
        f=0. if hi==lo else (t-times[lo])/(times[hi]-times[lo])
        def mix(key):return self.frames[key][lo]*(1-f)+self.frames[key][hi]*f
        # Heading is measured from actual skinned head minus tail, not rig names.
        a=self.box_yaw;rotation=np.asarray([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
        desired=rotation@np.array([1.,0.]);anatomy=mix('anatomy');source_heading=anatomy[0,:2]-anatomy[1,:2];source_heading/=np.linalg.norm(source_heading)
        yaw=math.atan2(desired[1],desired[0])-math.atan2(source_heading[1],source_heading[0])
        rr2=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
        dot=float((rr2@source_heading)@desired);self.heading_dot_min=min(self.heading_dot_min,dot)
        if dot<.999:raise RuntimeError('cat head points against entry trajectory')
        phase=float(np.clip((t-.20)/.68,0,1));q=phase*phase*(3-2*phase)
        root=np.r_[self.box_xyz[:2]-desired*.53*(1-q),self.box_xyz[2]+.0002+.012*q+.24*math.sin(math.pi*phase)]
        d.mocap_pos[self.mocap_id]=root;d.mocap_quat[self.mocap_id]=yaw_quat(math.degrees(yaw))
        endpoints=mix('endpoints');footboxes=mix('footboxes');capsules=[s for s in self.segments if s['kind']=='capsule'];nc=len(capsules)
        bounds=[]
        for index,(gid,s) in enumerate(zip(self.collider_ids,self.segments)):
            m.geom_sameframe[gid]=0
            if s['kind']=='capsule':
                aa,bb=endpoints[index];length=np.linalg.norm(bb-aa);r=s['radius_m']
                m.geom_pos[gid]=(aa+bb)/2;m.geom_quat[gid]=capsule_quat(bb-aa);m.geom_size[gid]=[r,length/2,0]
                m.geom_aabb[gid,:3]=0;m.geom_aabb[gid,3:]=[r,r,length/2+r];m.geom_rbound[gid]=length/2+r
                bounds.append((np.minimum(aa,bb)-r,np.maximum(aa,bb)+r))
            else:
                center,size=footboxes[index-nc];m.geom_pos[gid]=center;m.geom_quat[gid]=[1,0,0,0];m.geom_size[gid]=size
                m.geom_aabb[gid,:3]=0;m.geom_aabb[gid,3:]=size;m.geom_rbound[gid]=np.linalg.norm(size)
                bounds.append((center-size,center+size))
        # Exact animated mixed capsule/paw BVH bounds in body inertial frame.
        bid=m.body_name2id('e2_cat_root');inertial_rotation=np.empty(9);mujoco.mju_quat2Mat(inertial_rotation,m.body_iquat[bid]);inertial_rotation=inertial_rotation.reshape(3,3)
        corners=np.array([[[lo[x],lo[y],lo[z]]for x,y,z in []]for lo,hi in []]) if False else None
        transformed=[]
        for low,high in bounds:
            points=np.array([[xx,yy,zz]for xx in(low[0],high[0])for yy in(low[1],high[1])for zz in(low[2],high[2])])
            points=(points-m.body_ipos[bid])@inertial_rotation;transformed.append((points.min(0),points.max(0)))
        lows=np.array([x[0]for x in transformed]);highs=np.array([x[1]for x in transformed]);by_geom={g:i for i,g in enumerate(self.collider_ids)}
        adr=int(m.body_bvhadr[bid]);count=int(m.body_bvhnum[bid])
        for node in reversed(range(count)):
            absolute=adr+node;gid=int(m.bvh_nodeid[absolute])
            if gid>=0:low,high=lows[by_geom[gid]],highs[by_geom[gid]]
            else:
                children=[adr+int(x)for x in m.bvh_child[absolute]if x>=0];boxes=m.bvh_aabb[children]
                low=np.min(boxes[:,:3]-boxes[:,3:],axis=0);high=np.max(boxes[:,:3]+boxes[:,3:],axis=0)
            m.bvh_aabb[absolute,:3]=(low+high)/2;m.bvh_aabb[absolute,3:]=(high-low)/2
        bb=m.bvh_aabb[adr];under=max(float(np.max(bb[:3]-bb[3:]-lows)),float(np.max(highs-bb[:3]-bb[3:])),0)
        self.proxy_bvh_max_undercoverage_m=max(self.proxy_bvh_max_undercoverage_m,under)
        if under>1e-9:raise RuntimeError('cat BVH excludes collision proxies')
        for index,(mid,meshadr,n,affine) in enumerate(self.meshes):
            m.mesh_vert[meshadr:meshadr+n]=np.column_stack((mix(f'vertices_{index}'),np.ones(n)))@affine
            context=sim._render_context_offscreen
            if context is not None:
                v=m.mesh_vert[meshadr:meshadr+n];faces=self.frames[f'faces_{index}'];tri=v[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);vn=np.zeros_like(v)
                for corner in range(3):np.add.at(vn,faces[:,corner],normal)
                vn/=np.maximum(np.linalg.norm(vn,axis=1)[:,None],1e-12);normaladr=int(m.mesh_normaladr[mid]);m.mesh_normal[normaladr:normaladr+n]=vn
                context.gl_ctx.make_current();mujoco.mjr_uploadMesh(m._model,context.con,mid)
        sim.forward();rr=np.asarray(d.body_xmat[bid]).reshape(3,3)
        expected=root+m.geom_pos[self.collider_ids]@rr.T
        error=float(np.linalg.norm(expected-d.geom_xpos[self.collider_ids],axis=1).max());self.proxy_alignment_max_error_m=max(self.proxy_alignment_max_error_m,error)
        axes=(endpoints[:,1]-endpoints[:,0])@rr.T;axes/=np.linalg.norm(axes,axis=1)[:,None]
        actual=np.asarray(d.geom_xmat[self.collider_ids[:nc]]).reshape(-1,3,3)[:,:,2]
        axiserr=float(np.linalg.norm(axes-actual,axis=1).max());self.proxy_axis_max_error=max(self.proxy_axis_max_error,axiserr)
        if error>1e-8 or axiserr>1e-8:raise RuntimeError('cat collision proxy did not follow visual skeleton')
        self.current_visual_parts=[]
        for index,(mid,meshadr,n,affine)in enumerate(self.meshes):
            gid=m.geom_name2id(f'e2_cat_visual_{index}');gr=np.asarray(d.geom_xmat[gid]).reshape(3,3)
            actualworld=m.mesh_vert[meshadr:meshadr+n]@gr.T+d.geom_xpos[gid]
            sourceworld=mix(f'vertices_{index}')@rr.T+root
            self.visual_world_max_error_m=max(self.visual_world_max_error_m,float(np.linalg.norm(actualworld-sourceworld,axis=1).max()))
            self.current_visual_parts.append(actualworld)
        if self.visual_world_max_error_m>2e-6:raise RuntimeError('compiled mesh world pose differs from actual source skin')
        pawids=[g for g,s in zip(self.collider_ids,self.segments) if s['is_paw']]
        paw=np.mean(d.geom_xpos[pawids],axis=0);floor=float(object_bounds(sim,['box_floor'])[1][2]);local=rotation.T@(paw[:2]-self.box_xyz[:2]);point=np.r_[self.box_xyz[:2]+local,paw[2]]
        self.pet_in_target=hand_inside_target(point,self.box_xyz,floor,(.255,.155));self.pet_reached=t>=1
