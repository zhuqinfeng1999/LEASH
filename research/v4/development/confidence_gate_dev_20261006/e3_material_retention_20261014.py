"""E3-only common B0/Ours material-contact retention, private physics only.

Keep the existing authored-zone assist first. A completed actual CLOSE may
additionally retain the registered object if BOTH real finger collision sides
contact it and their contact midpoint satisfies the original between-pad AND
15/15/20 mm gripper-local test. No larger tolerance, automatic descent, pose
reset, object identity input, or fixed world-upright constraint is introduced.
The original weld captures the actual hand/object pose and OPEN detaches it.
"""
from itertools import product
from math import isfinite
from types import SimpleNamespace
import numpy as np
from research.v4.accelerated.grasp_assist import AssistEvent
VERSION='E3-common-post-close-material-retention-dev2-collision-name-20261014'
from research.v4.development.confidence_gate_dev_20261006.e2_physical_lift_trigger import (
    FINGER, FINGER_CONTACT_MAX_DISTANCE_M as CONTACT_MAX_DISTANCE_M)
# Reuse the actual existing compiled Panda right-gripper name/physical tolerance.

class MaterialRetentionAssist:
    def __init__(self,original):
        self.original=original;self.last_match=None
    def __getattr__(self,name):return getattr(self.original,name)
    @property
    def held(self):return self.original.held
    def reset(self):self.last_match=None;return self.original.reset()
    def before_control(self,command,*,source=''):
        # Preserves first-path authored eligibility and negative OPEN detach.
        return self.original.before_control(command,source=source)
    def _object_owner(self,geom_id):
        model=self.original.model;body=int(model.geom_bodyid[geom_id])
        visited=set()
        while body and body not in visited:
            visited.add(body)
            for i,(_zone,owner,_site,_weld)in enumerate(self.original._entries):
                if body==owner:return i
            body=int(model.body_parentid[body])
        return None
    def _midpoint_eligible(self,point,tolerance):
        data=self.original.data
        left=np.asarray(data.site_xpos[self.original._left]);right=np.asarray(data.site_xpos[self.original._right]);delta=right-left;gap=float(np.linalg.norm(delta))
        if not isfinite(gap)or gap<=1e-6:return False,None
        point=np.asarray(point,float)
        if point.shape!=(3,)or not np.isfinite(point).all():return False,None
        along=float((point-left)@(delta/gap));center=np.asarray(data.site_xpos[self.original._center]);frame=np.asarray(data.site_xmat[self.original._center]).reshape(3,3);local=frame.T@(point-center)
        valid=bool(0<=along<=gap and np.all(np.abs(local)<=np.asarray(tolerance)))
        return valid,dict(midpoint_world_m=point.tolist(),gripper_local_m=local.tolist(),along_pad_m=along,gap_m=gap,original_tolerance_xyz_m=list(tolerance))
    def after_executed_control(self,command):
        command=float(command)
        if not isfinite(command):raise ValueError('Executed gripper command must be finite')
        if command<=0 or self.original.held:return AssistEvent('unchanged')
        model,data=self.original.model,self.original.data;sides={}
        for c in data.contact[:data.ncon]:
            dist=float(c.dist)
            if not isfinite(dist)or dist>CONTACT_MAX_DISTANCE_M:continue
            a,b=int(c.geom1),int(c.geom2)
            for finger_id,other_id in((a,b),(b,a)):
                name=model.geom(finger_id).name or'';match=FINGER.fullmatch(name)
                if not match:continue
                index=self._object_owner(other_id)
                if index is None:continue
                pos=np.asarray(c.pos,float)
                if not np.isfinite(pos).all():continue
                sides.setdefault(index,{1:[],2:[]})[int(match.group(1))].append(dict(geom=name,item_geom=model.geom(other_id).name,position_world_m=pos.tolist(),contact_distance_m=dist))
        candidates=[]
        for index,contacts in sides.items():
            zone,*_=self.original._entries[index]
            for first,second in product(contacts[1],contacts[2]):
                midpoint=(np.asarray(first['position_world_m'])+np.asarray(second['position_world_m']))/2
                eligible,witness=self._midpoint_eligible(midpoint,zone.tolerance_xyz)
                if eligible:
                    candidates.append((index,dict(**witness,finger1_contact=first,finger2_contact=second)));break
        unique={i for i,_ in candidates}
        if len(unique)!=1:return AssistEvent('ambiguous'if unique else'no_match',len(unique))
        index,witness=candidates[0]
        self.original._attach(index) # Exact current relative pose, no teleport.
        self.last_match=dict(version=VERSION,actual_executed_command=command,qualification='real_two-finger_material_contact_and_original_between_pad_local_bounds',**witness)
        return AssistEvent('attached',1)


def private_evidence(scene):
    assist=scene.grasp_assist;original=assist.original;held=original._held_index
    row=dict(version=VERSION,assist_held=bool(assist.held),material_contact_match=assist.last_match,
             real_gripper_command=float(scene.last_action[6]),sequence=scene.sequence)
    if held is not None:
        _zone,body,_site,weld=original._entries[held];data=original.data
        hand_r=np.asarray(data.xmat[original._hand]).reshape(3,3);obj_r=np.asarray(data.xmat[body]).reshape(3,3)
        row.update(weld_active=bool(data.eq_active[weld]),actual_hand_world_rotation=hand_r.tolist(),actual_item_world_rotation=obj_r.tolist(),actual_hand_to_item_rotation=(hand_r.T@obj_r).tolist(),actual_hand_to_item_position_m=(hand_r.T@(np.asarray(data.xpos[body])-np.asarray(data.xpos[original._hand]))).tolist(),frozen_weld_relative_position_m=original.model.eq_data[weld,3:6].tolist(),frozen_weld_relative_quaternion_wxyz=original.model.eq_data[weld,6:10].tolist())
    return row


def configure(configured):
    if not configured.binding.instance.startswith('E3-'):return configured
    parent_scene,parent_evaluator=configured.Scene,configured.Evaluator
    class E3MaterialRetainedScene(parent_scene):
        shared_grasp_retention_version=VERSION
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.grasp_assist=MaterialRetentionAssist(self.grasp_assist)
        def step(self,action):
            result=super().step(action) # Failure/BLOCK has no returned execution.
            event=self.grasp_assist.after_executed_control(float(action[6]))
            if event.kind not in{'unchanged','no_match'}:
                self.assist_events.append(dict(sim_time_s=float(self.backend.sim.data.time),sequence=self.sequence,kind=event.kind,candidate_count=event.candidate_count,source=VERSION,executed_control_confirmed=True))
            return result
    class E3MaterialRetentionEvaluator(parent_evaluator):
        def observe(self,scene):
            row=super().observe(scene);row['private_e3_grasp_retention']=private_evidence(scene);return row
    return SimpleNamespace(**(vars(configured)|dict(Scene=E3MaterialRetainedScene,Evaluator=E3MaterialRetentionEvaluator,shared_grasp_retention_version=VERSION)))
