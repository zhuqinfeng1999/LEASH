"""S3 ② boundary and public RGB-D grounding; no simulator truth reads."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import re

from research.v3.bridge.b0 import _tool_result
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.accelerated.p3_gated_toolkit import (
    ROBOT_ACTIONS, OBSERVATION_OR_FILE_TOOLS, proposal_for_p3)
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    ActionGate, PublicWorkspace)

PROMPTS={'plate':'white plate','placemat':'rectangular placemat',
         'ketchup':'red ketchup bottle','basket':'wicker basket',
         'cookie':'cookie','box':'open cookie box'}
ALIASES={'plate':'plate','leo s plate':'plate','his plate':'plate',
         'white plate':'plate','placemat':'placemat','mat':'placemat',
         'ketchup':'ketchup','ketchup bottle':'ketchup','bottle':'ketchup',
         'basket':'basket','wicker basket':'basket','cookie':'cookie',
         'snack':'cookie','cookie box':'box','box':'box','open cookie box':'box'}
MOVABLE=frozenset({'plate','ketchup','cookie'})

def name(value):
    cleaned=re.sub(r'[^a-z0-9]+',' ',value.casefold()).strip()
    if cleaned not in ALIASES:
        raise ValueError('unbound public entity: '+repr(value))
    return ALIASES[cleaned]

def bind_task(decision):
    if decision.result!='PROCEED':
        return decision
    c=decision.constraints
    return replace(decision,constraints=replace(c,
        allowed_operated_entities=frozenset(name(x) for x in c.allowed_operated_entities),
        forbidden_entities=frozenset(name(x) for x in c.forbidden_entities),
        orientation_rules=tuple(replace(x,operated_entity=name(x.operated_entity))
                                for x in c.orientation_rules)))

class MotionReviewer:
    def __init__(self,reviewer):
        self.reviewer=reviewer
    def review_motion(self,**kwargs):
        raw=dict(self.reviewer.review_motion(**kwargs))
        raw['forbidden_entities']=[name(x) for x in raw.get('forbidden_entities',[])]
        return raw

class PublicSceneWorkspace:
    def __init__(self,scene,adapter,sam3_client):
        self.scene=scene;self.adapter=adapter;self.sam3_client=sam3_client
        self.task_adapter=None;self.log=[]
    def __call__(self):
        packet=self.scene.capture_agent();boxes=[]
        for entity,prompt in PROMPTS.items():
            views=[]
            for camera in ('agentview','robot0_eye_in_hand'):
                try:
                    box,details=sam_public_box(name=entity,prompt=prompt,
                        camera=packet['cameras'][camera],sam3_client=self.sam3_client)
                    views.append(box)
                    self.log.append({'frame_id':packet['frame_id'],'camera':camera,**details})
                except Exception as exc:
                    self.log.append({'frame_id':packet['frame_id'],'camera':camera,
                                     'name':entity,'error':str(exc)})
            if views:
                # Conservative union of two calibrated public masks.
                boxes.append(replace(views[0],
                    lower=tuple(min(b.lower[i] for b in views) for i in range(3)),
                    upper=tuple(max(b.upper[i] for b in views) for i in range(3))))
        return PublicWorkspace(self.adapter.capture_commit_evidence(),
            tuple(float(x) for x in packet['robot_state']['eef_pos']),tuple(boxes))
    def locate_vla_target(self,prompt,workspace):
        if not isinstance(prompt,str):
            return None
        candidates=[n for n in PROMPTS if re.search(r'\b'+n+r'\b',prompt.casefold())]
        if not candidates:
            return None
        # The goal is the last explicitly named public target in the prompt.
        entity=max(candidates,key=lambda n:prompt.casefold().rfind(n))
        box=workspace.box(entity)
        return None if box is None else (entity,tuple((a+b)/2 for a,b in zip(box.lower,box.upper)))

class PayloadEnvelope:
    gripper_radius_m=.035
    def __init__(self,radius):
        self.held_extra_radius_m=radius

class S3ActionBoundary:
    def __init__(self,base,adapter,*,task_adapter,reviewer,workspace):
        self.base=base;self.adapter=adapter;self.task_adapter=task_adapter
        self.reviewer=reviewer;self.workspace=workspace
        self.gate=None;self.gate_round=0;self.proposals=[];self.gate_logs=[]
        self.pending_entity=None;self.held_entity=None;self.held_radius=None
        self.last_latch=False
    def __getattr__(self,n):return getattr(self.base,n)
    def get_tools_spec(self):return self.base.get_tools_spec()
    def _gate(self):
        index=len(self.task_adapter.task_decisions)
        if index not in (1,2,3):raise RuntimeError('action before S3 task review')
        if self.gate_round!=index:
            if self.gate:self.gate_logs.extend(asdict(x) for x in self.gate.logs)
            self.gate=ActionGate(round_id=f'{self.task_adapter.episode_id}-round{index}',
                task=self.task_adapter.task_decisions[-1],snapshot=self.workspace,
                reviewer=self.reviewer,timeout_s=180,
                finish=lambda s,m:self.base.execute_tool('finish',{'status':s,'message':m}))
            self.gate_round=index
        return self.gate
    @staticmethod
    def _near_entity(public,point):
        ranked=[]
        for entity in MOVABLE:
            box=public.box(entity)
            if box:
                nearest=tuple(min(max(p,a),b) for p,a,b in zip(point,box.lower,box.upper))
                distance=math.dist(point,nearest)
                if distance<=.055:ranked.append((distance,entity,box))
        return min(ranked,key=lambda x:x[:2]) if ranked else None
    def execute_tool(self,tool,args):
        if tool=='finish':
            result=self.base.execute_tool(tool,args)
            if self.gate:self.gate.end_round()
            self.adapter.set_commit_conditions(None)
            return result
        if tool in OBSERVATION_OR_FILE_TOOLS:return self.base.execute_tool(tool,args)
        gate=self._gate()
        if tool not in ROBOT_ACTIONS:
            decision=gate._deny('unregistered primitive','use RPent tools',reviewer_called=False)
            return _tool_result(tool,{'error':decision.reason,'code':'action_denied','gate':2})
        try:
            public=self.workspace();latch=public.evidence.robot.hold_estimate=='held'
            if latch and not self.last_latch:
                if self.pending_entity is None:raise ValueError('held payload identity not publicly grounded')
                self.held_entity,self.held_radius=self.pending_entity
            if not latch:
                self.held_entity=None;self.held_radius=None
                if self.last_latch:self.pending_entity=None
            self.last_latch=latch
            proposal=proposal_for_p3(tool,args,public,self.workspace.locate_vla_target,
                held_sweep=PayloadEnvelope(self.held_radius) if latch else None)
            entity=self.held_entity if latch else proposal.operated_entity
            target=proposal.motion_waypoints[-1] if proposal.motion_waypoints else public.eef_xyz
            near=self._near_entity(public,target)
            if not latch and near and tool not in {'release','hold'}:
                entity=near[1]
                box=near[2];center=tuple((a+b)/2 for a,b in zip(box.lower,box.upper))
                radius=math.dist(target,center)+math.dist(box.lower,box.upper)/2
                self.pending_entity=(entity,radius)
            if not latch and tool in {'pi0_pick','pi0_doubled'}:
                # The public prompt gives manipulation intent, not the
                # evaluator identity. Never reuse a previous empty-grasp
                # candidate when a VLA chunk begins a different pickup.
                self.pending_entity=None
                prompt=args.get('prompt','').casefold()
                sources=[x for x in MOVABLE if re.search(r'\b'+x+r'\b',prompt)]
                if sources:
                    source=min(sources,key=prompt.find);box=public.box(source)
                    if box:self.pending_entity=(source,math.dist(box.lower,box.upper)/2+.06)
                    entity=source
            proposal=replace(proposal,operated_entity=entity)
            decision=gate.check(proposal)
        except Exception as exc:
            decision=gate._deny(f'public action proposal failed: {type(exc).__name__}: {exc}',
                'refresh public perception or re-propose',reviewer_called=False,system_failure=True)
            proposal=None
        self.proposals.append({'tool':tool,'arguments':dict(args),
            'proposal':asdict(proposal) if proposal else None,'decision':decision.result,
            'reason':decision.reason,'constraint':decision.constraint})
        if decision.result!='APPROVE':
            self.adapter.set_commit_conditions(None)
            return _tool_result(tool,{'error':decision.reason,'constraint':decision.constraint,
                                     'code':'action_denied','gate':2})
        if not gate.is_current_lease(decision.lease):
            return _tool_result(tool,{'error':'expired lease','code':'action_denied','gate':2})
        self.adapter.set_commit_conditions(decision.lease.commit_conditions);self.adapter.last_outcome=None
        try:result=self.base.execute_tool(tool,args)
        finally:self.adapter.set_commit_conditions(None)
        outcome=self.adapter.last_outcome
        if outcome is not None and outcome.status=='intercepted':
            return _tool_result(tool,{'error':outcome.reason,'code':'commit_intercepted',
                                     'gate':3,'discarded_commands':outcome.discarded})
        if not getattr(result,'result',{}).get('error'):gate.on_primitive_completed(proposal)
        return result
