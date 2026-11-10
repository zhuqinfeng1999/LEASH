"""Original constructor + one fixed common 80-control reset; no task macro."""
import json
import time
from pathlib import Path
from types import FunctionType
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.r302_front_slot_short_v_20260928 import scene as parent
from research.v4.candidates.r302_front_slot_short_v_20260928.support import observed_sample as old_sample
from .admit import verify

H=Path(__file__).resolve().parent
_RAISED_INIT=FunctionType(parent._RAISED_INIT.__code__,dict(parent._RAISED_INIT.__globals__,H=H),
                         closure=parent._RAISED_INIT.__closure__)
_FRONT_INIT=FunctionType(parent.FrontSlotShortVCanScene.__init__.__code__,
                        dict(parent.FrontSlotShortVCanScene.__init__.__globals__,H=H,_RAISED_INIT=_RAISED_INIT),
                        closure=parent.FrontSlotShortVCanScene.__init__.__closure__)
sample=FunctionType(old_sample.__code__,dict(old_sample.__globals__,H=H))

class Fixed80FrontSlotCanScene(parent.FrontSlotShortVCanScene):
    layout_version='R302-same-front-shortV-raised-fixed80-open-reset-v1'
    original_total_control_cap=1200
    original_total_wall_cap_s=90.
    startup_control_count=80
    def __init__(self,*args,**kwargs):
        verify();self.episode_started_monotonic=time.monotonic()
        out=H/'startup_seed0';out.mkdir()
        _FRONT_INIT(self,*args,**kwargs)
        frames=[]
        try:
            frames.append(sample(self))
            for _ in range(self.startup_control_count):
                self.step(np.asarray([0.,0.,0.,0.,0.,0.,-1.]))
                row=sample(self)
                m,d=self.backend.sim.model._model,self.backend.sim.data._data
                f=[int(m.jnt_qposadr[m.joint(n).id]) for n in ('gripper0_right_finger_joint1','gripper0_right_finger_joint2')]
                row['actual_finger_qpos_m']=d.qpos[f].tolist()
                row['actual_assist_held']=bool(self.grasp_assist.held)
                frames.append(row)
            tail=frames[-11:];linear=np.asarray([np.linalg.norm(s['can_qvel'][:3]) for s in tail]);angular=np.asarray([np.linalg.norm(s['can_qvel'][3:]) for s in tail])
            world_forces=[]
            for s in tail:
                total=np.zeros(3);details=[]
                for c in s['actual_all_contact_forces']:
                    a,b=c['geoms'];sign=-1 if a.startswith('steel_thermos_1_') else 1 if b.startswith('steel_thermos_1_') else 0
                    if sign:
                        F=sign*np.asarray(c['contact_frame_world']).reshape(3,3).T@np.asarray(c['force_contact_frame'][:3]);total+=F
                        details.append(dict(c,force_on_can_world_N=F.tolist()))
                world_forces.append(total.tolist());s['can_world_support_contacts']=details;s['total_can_world_contact_force_N']=total.tolist()
            weight=.4*abs(float(self.backend.sim.model._model.opt.gravity[2]));forces=np.asarray(world_forces)
            xyz=np.asarray([s['can_qpos'][:3] for s in tail]);R=Rotation.from_quat(np.asarray([s['can_qpos'][3:] for s in tail])[:,[1,2,3,0]])
            initial=np.asarray(frames[0]['can_qpos'][:2])
            stable=bool(np.all(linear<=.02) and np.all(angular<=.1)
                        and np.all(forces[:,2]>=.75*weight) and np.all(forces[:,2]<=1.25*weight)
                        and max(np.linalg.norm(np.asarray(s['can_qpos'][:2])-initial) for s in tail)<.005
                        and all(s['actual_assist_held'] is False for s in tail))
            report={'kind':'ONE actual fixed80 open-neutral common initialization',
              'passed':stable,'neutral_controls':80,'task_controls':0,'source_sequence_after_startup':self.sequence,
              'total_control_cap_unchanged':1200,'task_controls_remaining':1200-self.sequence,
              'total_wall_cap_s_unchanged':90.,'startup_elapsed_wall_s':time.monotonic()-self.episode_started_monotonic,
              'open_neutral_command':[0.,0.,0.,0.,0.,0.,-1.],'fixed_not_wait_until_stable':True,
              'all_frames':frames,'each20_checkpoints':[frames[i] for i in (0,20,40,60,80)],
              'last_half_second_frames':tail,'last_half_second_linear_speed_range_m_s':[float(linear.min()),float(linear.max())],
              'last_half_second_angular_speed_range_rad_s':[float(angular.min()),float(angular.max())],
              'last_half_second_XYZ_peak_to_peak_m':np.ptp(xyz,axis=0).tolist(),
              'last_half_second_orientation_from_start_max_deg':float(np.rad2deg((R*R[0].inv()).magnitude()).max()),
              'last_half_second_world_support_Z_range_N':[float(forces[:,2].min()),float(forces[:,2].max())],
              'weight_N':weight,'support_not_dual_rail_requirement':True,'actual_bind_or_task_claim':False,
              'GPU_models':0}
            m,d=self.backend.sim.model._model,self.backend.sim.data._data
            mujoco.mj_saveModel(m,str(out/'compiled_model.mjb'))
            np.savez_compressed(out/'constructor_data.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),time=np.asarray(d.time))
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if not stable:raise RuntimeError('First actual fixed80 final0.5s support/rest evidence failed; STOP')
        except BaseException as exc:
            (out/'error.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'actual_sequence':self.sequence,'saved_frames':frames},indent=2)+'\n')
            self.close();raise

    def step(self,action):
        if self.sequence>=self.original_total_control_cap:raise RuntimeError('Original1200 total controls exhausted including startup')
        if time.monotonic()-self.episode_started_monotonic>=self.original_total_wall_cap_s:raise RuntimeError('Original90s total wall exhausted including startup')
        return super().step(action)
