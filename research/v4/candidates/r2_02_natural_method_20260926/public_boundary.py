"""T1 noun adapter over the tested public payload/action/commit machinery.

The legacy payload helper internally looks up 'iron'; ItemWorkspace maps that
implementation lookup to the actual publicly grounded item. Neither reviewer
nor planner receives an iron identity, private region, or evaluator metadata.
"""
from dataclasses import replace
import re
from unittest.mock import patch
from research.v4.candidates.p4_natural_method_20260926 import public_boundary as payload
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicWorkspace

PUBLIC_NOUNS={'item':'incense holder','box':'cardboard box'}
ALIASES={'item':'item','incense':'item','incense holder':'item','ash catcher':'item',
    'ashcatcher':'item','lit incense':'item','burning incense':'item','incense stick':'item',
    'box':'box','cardboard box':'box','paper box':'box','container':'box'}
def bind_name(value):
    if not isinstance(value,str):raise ValueError('public entity name must be text')
    key=re.sub(r'[^a-z0-9]+',' ',value.casefold()).strip()
    if key not in ALIASES:raise ValueError('unbound T1 public entity: '+repr(value))
    return ALIASES[key]
def bind_task(decision):
    if decision.result!='PROCEED':return decision
    c=decision.constraints
    return replace(decision,constraints=replace(c,
        allowed_operated_entities=frozenset(bind_name(x) for x in c.allowed_operated_entities),
        forbidden_entities=frozenset(bind_name(x) for x in c.forbidden_entities),
        orientation_rules=tuple(replace(x,operated_entity=bind_name(x.operated_entity)) for x in c.orientation_rules)))

class ItemWorkspace(PublicWorkspace):
    def box(self,name):return super().box('item' if name=='iron' else name)

class PublicWorkspaceReader(payload.PublicWorkspaceReader):
    def __call__(self):
        start=len(self.log)
        # Legacy 'iron' also marks its movable payload as never cached. Only
        # the object noun changes; pixels, thresholds/calibration stay shared.
        with patch.object(payload,'PROMPTS',{'iron':PUBLIC_NOUNS['item'],'box':PUBLIC_NOUNS['box']}):
            original=super().__call__()
        for row in self.log[start:]:
            if row.get('name')=='iron':row['name']='item'
        boxes=tuple(replace(b,name='item') if b.name=='iron' else b for b in original.visible_entities)
        return ItemWorkspace(original.evidence,original.eef_xyz,boxes)
    def locate_vla_target(self,prompt,workspace):
        if not isinstance(prompt,str):return None
        hits=[(prompt.casefold().rfind(name),entity) for name,entity in ALIASES.items()
              if re.search(r'\b'+re.escape(name)+r'\b',prompt.casefold())]
        if not hits:return None
        entity=max(hits)[1];box=workspace.box(entity)
        return None if box is None else (entity,tuple((a+b)/2 for a,b in zip(box.lower,box.upper)))

class PublicMotionReviewer(payload.PublicMotionReviewer):
    def review_motion(self,*,request,workspace,lease,deadline_monotonic_ns):
        raw=dict(self.session.review_motion(request=request,workspace=self.public_workspace or workspace,
            lease=lease,deadline_monotonic_ns=deadline_monotonic_ns))
        raw['forbidden_entities']=[bind_name(x) for x in raw.get('forbidden_entities',[])]
        return raw

class T1ActionBoundary(payload.P4ActionBoundary):
    def _proposal(self,name,arguments,workspace,locate):
        proposal=super()._proposal(name,arguments,workspace,locate)
        return replace(proposal,operated_entity='item' if proposal.operated_entity=='iron' else proposal.operated_entity)
    def _missing_payload(self,name):
        from research.v3.bridge.b0 import _tool_result
        decision=self.gate._deny('held item envelope not grounded in public RGB-D',
            'refresh public perception before transport',reviewer_called=False)
        self.proposals.append({'tool':name,'decision':decision.result,'reason':decision.reason})
        return _tool_result(name,{'error':decision.reason,'code':'action_denied','gate':2,'constraint':decision.constraint})
