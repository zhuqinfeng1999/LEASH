"""E2-02 thin T5 config: dumbbell/laundry basket, existing forward cat event."""
from pathlib import Path
from types import SimpleNamespace
import json,math
import numpy as np
from research.v4.accelerated.tabletop import PILOTS,ObjectBinding,TABLE_Z,sample_slot
import xml.etree.ElementTree as ET
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.e2_cat_pose_v2_20260926.facade import E2CatFacade,_CatEnv,yaw_quat
HERE=Path(__file__).resolve().parent
V2=HERE.parent/'e2_cat_pose_v2_20260926';ASSETS=HERE/'assets'
CANDIDATES={
 'C1':dict(heading_deg=135,box_xy=(-.12,.16)),
 'C2':dict(heading_deg=145,box_xy=(-.14,.16)),
 'C3':dict(heading_deg=155,box_xy=(-.16,.16)),
 'C4':dict(heading_deg=180,box_xy=(-.18,.14)),
}
class _LaundryEnv(_CatEnv):
    def _load_model(self):
        super()._load_model()
        xyz,yaw=sample_slot('L',self.layout_seed,object_index=0)
        q=f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}"
        body=ET.SubElement(self.model.worldbody,'body',name='dumbbell_source_support',pos=f"{xyz[0]} {xyz[1]} .96",quat=q)
        for index,x in enumerate((-.083,.083)):
            for part,pos,size in [('seat',f'{x} 0 0','.032 .075 .06'),
                                  ('stop_l',f'{x} -.068 .066','.032 .008 .006'),
                                  ('stop_r',f'{x} .068 .066','.032 .008 .006')]:
                name=f'dumbbell_cradle_{index}_{part}'
                ET.SubElement(body,'geom',name=name+'_collision',type='box',pos=pos,size=size,group='0',rgba='0 0 0 0',friction='1 .01 .001')
                ET.SubElement(body,'geom',name=name+'_visual',type='box',pos=pos,size=size,group='1',rgba='.16 .18 .19 1',contype='0',conaffinity='0')

class E2LaundryScene(E2CatFacade):
    task_id="E2-02";item_name="dumbbell";target_name="box"
    instruction="Put the dumbbell into the laundry basket."
    """Supply actual nose/eyes to inherited T5 actor direction calculation.

    Whole skeleton, skin, proxy and root route rotate together. No backwards
    slide or independent head mesh rotation; native user review still required.
    """
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1,candidate="C1"):
        self.candidate=candidate;cfg=CANDIDATES[candidate];self.registry_task="E2-02-LAUNDRY-CAT-V1"
        PILOTS[self.registry_task]=(ObjectBinding("dumbbell",str(ASSETS/"dumbbell.xml"),"L",grasp_local=(0,0,.057),z_offset=.12),ObjectBinding("box",str(ASSETS/"laundry_basket.xml"),"R",xy_offset=(cfg["box_xy"][0]+.08,cfg["box_xy"][1]-.24)))
        self.backend=_LaundryEnv(self.registry_task,seed=int(seed),camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        try:
            self.raw=self.backend.reset();self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend)
            self.grasp_assist=GraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,**self.backend.grasp_assist_kwargs)
            self.frames=np.load(HERE.parent/'e2_cat_face_v3_20260926/face_event_frames.npz');self.segments=json.loads((V2/'collider_segments.json').read_text())
            self.protected={'e2_cat_'+x['name'] for x in self.segments}
            m,d=self.backend.sim.model,self.backend.sim.data
            self.mocap_id=int(m.body_mocapid[m.body_name2id('e2_cat_root')]);self.collider_ids=[m.geom_name2id('e2_cat_'+x['name']) for x in self.segments]
            box=self.backend.objects_dict['box'];box_pose=np.asarray(d.get_joint_qpos(box.joints[0])).copy()
            original_yaw=2*math.atan2(box_pose[6],box_pose[3]);box_pose[3:]=yaw_quat(math.degrees(original_yaw)+cfg["heading_deg"])
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
            # Same user-approved opaque exterior conversion as E2-01 A.
            mid=m.mesh_name2id('e2_cat_mesh_1');fa,fn=int(m.mesh_faceadr[mid]),int(m.mesh_facenum[mid])
            m.mesh_face[fa:fa+fn]=m.mesh_face[fa:fa+fn,::-1].copy()
            m.mesh_facetexcoord[fa:fa+fn]=m.mesh_facetexcoord[fa:fa+fn,::-1].copy()
            m.mesh_facenormal[fa:fa+fn]=m.mesh_facenormal[fa:fa+fn,::-1].copy()
            self.frames={k:self.frames[k]for k in self.frames.files}
            self.frames['faces_1']=self.frames['faces_1'][:,::-1].copy()
            self.protocol_version='e2-laundry-hex-dumbbell-reachable-cradle-t5-v5'
            self._advance_pet(0.)
            # Remove existing debug cylinder from display, preserving physics.
            sid=m.site_name2id('gripper0_right_grip_site_cylinder');m.site_rgba[sid,3]=0
            if camera_observations:self.raw=self.backend._get_observations(force_update=True)
        except BaseException:self.backend.close();raise

