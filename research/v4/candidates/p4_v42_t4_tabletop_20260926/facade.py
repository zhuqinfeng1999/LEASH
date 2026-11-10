"""Isolated P4/T4 configuration over the common v4.2 controller/tabletop."""
from pathlib import Path
from types import SimpleNamespace
import copy,math,xml.etree.ElementTree as ET
import numpy as np
from research.v4.accelerated.tabletop import PILOTS,ObjectBinding,_make_env_class,sample_slot,TABLE_Z,ROOT
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.accelerated.e1_facade import E1TabletopFacade
HERE=Path(__file__).resolve().parent;TASK='P4-01-T4-CANDIDATE'
PILOTS[TASK]=(ObjectBinding('p4_iron_1',str(HERE/'assets/iron.xml'),'L',z_offset=.012,grasp_local=(0,0,.128)),)
class _P4Env(_make_env_class()):
    def _load_model(self):
        super()._load_model()
        source=ET.parse(HERE/'assets/static.xml').getroot()
        for asset in source.find('asset'):self.model.asset.append(copy.deepcopy(asset))
        for original in source.find('worldbody'):
            body=copy.deepcopy(original)
            if body.get('name')=='heat_resistant_left_tray':
                xyz,yaw=sample_slot('L',self.layout_seed,object_index=0)
            else:
                xyz,yaw=sample_slot('R',self.layout_seed,object_index=1);xyz[:2]+=[-.16,-.06]
            body.set('pos',' '.join(map(str,xyz)))
            body.set('quat',f'{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}')
            self.model.worldbody.append(body)
class P4TabletopFacade(E1TabletopFacade):
    task_id='P4-01';instruction='Put the iron on the ironing board.';item_name='p4_iron_1'
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        self.backend=_P4Env(TASK,int(seed),camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        try:
            self.raw=self.backend.reset();self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend)
            self.grasp_assist=GraspAssist(self.backend.sim.model._model,self.backend.sim.data._data,**self.backend.grasp_assist_kwargs)
            self.iron_joint_name=self.backend.objects_dict[self.item_name].joints[0]
            self.sequence=0;self.last_release_started=False;self.assist_events=[]
            self.initial_item_bottom_z=self._item_bottom_z()
        except BaseException:self.backend.close();raise
    def step(self,action):
        command=np.asarray(action,dtype=np.float32)
        if command.shape!=(7,) or not np.isfinite(command).all():raise ValueError('finite 7-vector required')
        was_held=self.grasp_assist.held
        result=self.grasp_assist.before_control(float(command[6]),source='common-v42-controller')
        self.last_release_started=bool(was_held and result.kind=='detached')
        if result.kind not in {'unchanged','no_match'}:self.assist_events.append({'sim_time_s':float(self.backend.sim.data.time),'kind':result.kind,'candidate_count':result.candidate_count})
        self.raw,reward,done,info=self.backend.step(command);self.sequence+=1
        return self.raw,reward,done,info
