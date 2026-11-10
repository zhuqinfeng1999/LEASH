"""Original P1 controller with one explicit normal hover release, offline only."""
import inspect
import json
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
from research.v4.candidates.p1_ma_fallback_20260925 import candidate as base
from research.v4.candidates.p1_04_cutter_holder_20260926.scene import P1CutterFacade
from research.v4.candidates.p1_04_cutter_holder_20260926.scorer import P1HolderEvaluator,holder_evidence
from .geometry import hover
HERE=Path(__file__).resolve().parent
TRACE=[];PROOFS=[];PREOPEN=[]

def snapshot(scene):
    m=scene.backend.sim.model._model;d=scene.backend.sim.data._data
    item=scene.backend.objects_dict[scene.item_name];target=scene.backend.objects_dict[scene.target_name]
    own={m.geom(n).id for n in item.contact_geoms};contacts=[];bad=[]
    for index,c in enumerate(d.contact[:d.ncon]):
        a,b=int(c.geom1),int(c.geom2);names=[m.geom(a).name or'',m.geom(b).name or'']
        force=np.zeros(6);mujoco.mj_contactForce(m,d,index,force)
        if a in own or b in own:
            contacts.append(dict(geoms=names,distance_m=float(c.dist),force6_local=force.tolist(),normal_world=np.asarray(c.frame).reshape(3,3)[0].tolist()))
        robot=[n.startswith(('robot0_','gripper0_'))for n in names]
        if sum(robot)==1 and not({a,b}&own)and c.dist<-.001:
            bad.append(dict(geoms=names,distance_m=float(c.dist),force6_local=force.tolist()))
    item_pose=np.asarray(scene.backend.sim.data.get_joint_qpos(item.joints[0])).copy()
    vel=np.asarray(scene.backend.sim.data.get_joint_qvel(item.joints[0])).copy()
    return dict(sequence=scene.sequence,time_s=float(d.time),eef_m=np.asarray(scene.raw['robot0_eef_pos']).tolist(),
        qpos=np.asarray(d.qpos).tolist(),qvel=np.asarray(d.qvel).tolist(),eef_quat_xyzw=np.asarray(scene.raw['robot0_eef_quat']).tolist(),item_pose=item_pose.tolist(),item_velocity6=vel.tolist(),linear_speed_m_s=float(np.linalg.norm(vel[:3])),
        angular_speed_rad_s=float(np.linalg.norm(vel[3:])),attached=bool(scene.grasp_assist.held),
        holder_geometry=holder_evidence(scene),item_contacts=contacts,deep_robot_environment_contacts=bad)

class RecordedScene(P1CutterFacade):
    def step(self,action):
        if float(action[6])<0 and self.grasp_assist.held:PREOPEN.append(snapshot(self))
        value=super().step(action);row=snapshot(self);row['action']=np.asarray(action,np.float32).tolist();TRACE.append(row)
        if row['deep_robot_environment_contacts']:raise RuntimeError('Actual macro robot/environment penetration; stopped')
        return value

def _hover_goal(scene):
    goal,proof=hover(scene);PROOFS.append(proof);return goal

def _preopen_check(scene,stages):
    row=snapshot(scene);g=row['holder_geometry'];stages.append(dict(stage='exact_actual_before_open',held=row['attached'],geometry=g,item_pose=row['item_pose'],tool_contacts=row['item_contacts']))
    if not row['attached']or not g['in_target']or row['deep_robot_environment_contacts']:
        raise RuntimeError('Actual pre-opening hold/full footprint/clearance not valid')

source=inspect.getsource(base.probe)
start=source.index('                plate_geom = next(name for name in target.contact_geoms')
end=source.index('                hold(-1., 24)',start)
source=source[:start]+'''                release_eef=_hover_goal(scene)
                move(release_eef,"explicit_hover_release",grip=1.,clip=.005,max_steps=180)
                _preopen_check(scene,stages)
'''+source[end:]
namespace={**vars(base),'P1CandidateFacade':RecordedScene,'P1CandidateEvaluator':P1HolderEvaluator,
           '_hover_goal':_hover_goal,'_preopen_check':_preopen_check}
exec(compile(source,__file__,'exec'),namespace)

def probe(mode,seed):
    TRACE.clear();PROOFS.clear();PREOPEN.clear()
    result=namespace['probe'](mode,seed)
    result.update(instance='P1-04',instruction=P1CutterFacade.instruction,hover_proofs=list(PROOFS),
        exact_preopen=list(PREOPEN),physical_trace=list(TRACE),kind='offline known-script explicit gravity placement; no model')
    if mode=='safe' and result.get('result'):
        evidence=result['result']['completion_evidence'];begin=evidence['rest_started_at_s'];end=evidence['rest_ended_at_s']
        window=[r for r in TRACE if begin is not None and end is not None and begin-1e-8<=r['time_s']<=end+1e-8]
        result['exact_supported_window']=window
    return result
