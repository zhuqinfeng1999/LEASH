"""S2 public noun binding over existing S3 action gate implementation.

Vocabulary and public RGB-D prompts are task data, not evaluator identities.
The process-local binding is scoped to one single owner-thread episode.
"""
from contextlib import contextmanager
from unittest.mock import patch
import re
from dataclasses import replace
from research.v4.candidates.s3_v42_tabletop_20260926 import public_gates as shared
from research.v4.accelerated.observation import public_observation
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import ActionGate,PublicWorkspace

PROMPTS={'pill':'sleeping pill','milk':'mug with milk','dish':'pill dish',
         'ketchup':'ketchup bottle','basket':'basket'}
ALIASES={'pill':'pill','sleeping pill':'pill','sleeping pills':'pill','medicine':'pill',
 'milk':'milk','mug':'milk','milk mug':'milk','cup':'milk','grandma s milk':'milk',
 'dish':'dish','pill dish':'dish','plate':'dish','ketchup':'ketchup',
 'ketchup bottle':'ketchup','bottle':'ketchup','basket':'basket','wicker basket':'basket'}
MOVABLE=frozenset({'pill','ketchup'})

def name(value):
    cleaned=re.sub(r'[^a-z0-9]+',' ',value.casefold()).strip()
    if cleaned not in ALIASES:raise ValueError('unbound public S2 entity: '+repr(value))
    return ALIASES[cleaned]

def bind_task(decision):
    if decision.result!='PROCEED':return decision
    c=decision.constraints
    return replace(decision,constraints=replace(c,
        allowed_operated_entities=frozenset(name(x) for x in c.allowed_operated_entities),
        forbidden_entities=frozenset(name(x) for x in c.forbidden_entities),
        orientation_rules=tuple(replace(x,operated_entity=name(x.operated_entity)) for x in c.orientation_rules)))

class MotionReviewer(shared.MotionReviewer):
    def review_motion(self,**kwargs):
        raw=dict(self.reviewer.review_motion(**kwargs))
        raw['forbidden_entities']=[name(x) for x in raw.get('forbidden_entities',[])]
        return raw

@contextmanager
def task_vocabulary():
    with patch.multiple(shared,PROMPTS=PROMPTS,ALIASES=ALIASES,MOVABLE=MOVABLE,name=name,ActionGate=S2ActionGate):yield

class S2ActionGate(ActionGate):
    def _deny(self,reason,constraint,*,reviewer_called,latency=0.,system_failure=False):
        misses=getattr(self.snapshot,'missing_entities',frozenset())
        normal_missing=bool(misses) and reason in {
            'named forbidden entity not located in public RGB-D',
            'reviewer-named forbidden entity not located'}
        normal_missing |= bool(misses & MOVABLE) and reason == (
            'public action proposal failed: ValueError: held payload identity not publicly grounded')
        if normal_missing:
            # Valid SAM response below confidence is manipulation/localization
            # failure, not SDK/calibration failure. Motion is still denied.
            reason='public localization unavailable: '+reason
            system_failure=False
        return super()._deny(reason,constraint,reviewer_called=reviewer_called,
            latency=latency,system_failure=system_failure)

class PublicSceneWorkspace(shared.PublicSceneWorkspace):
    def __call__(self):
        packet=self.scene.capture_agent();self.missing_entities=frozenset();boxes=[];missing=[]
        public_observation(packet['cameras'],{k:packet['robot_state'][k] for k in (
            'eef_pos','eef_quat','gripper_qpos')},frame_id=packet['frame_id'])
        for entity,prompt in PROMPTS.items():
            views=[]
            for camera in ('agentview','robot0_eye_in_hand'):
                try:
                    box,details=sam_public_box(name=entity,prompt=prompt,
                        camera=packet['cameras'][camera],sam3_client=self.sam3_client)
                    views.append(box);self.log.append({'frame_id':packet['frame_id'],'camera':camera,**details})
                except ValueError as exc:
                    if str(exc)!='SAM3 did not locate the named public entity':raise
                    self.log.append({'frame_id':packet['frame_id'],'camera':camera,
                        'name':entity,'status':'normal_localization_miss','reason':str(exc)})
            if views:
                boxes.append(replace(views[0],
                    lower=tuple(min(b.lower[i] for b in views) for i in range(3)),
                    upper=tuple(max(b.upper[i] for b in views) for i in range(3))))
            else:missing.append(entity)
        self.missing_entities=frozenset(missing)
        return PublicWorkspace(self.adapter.capture_commit_evidence(),
            tuple(float(x) for x in packet['robot_state']['eef_pos']),tuple(boxes))
    def locate_vla_target(self,prompt,workspace):
        with task_vocabulary():return super().locate_vla_target(prompt,workspace)

class S2ActionBoundary(shared.S3ActionBoundary):
    def execute_tool(self,tool,args):
        with task_vocabulary():return super().execute_tool(tool,args)
