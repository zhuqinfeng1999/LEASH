"""Only user-goal and public object/limb names; never inject printed text."""
import re
from dataclasses import replace
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.e2_laundry_method_20260926.public_intent import infer_public_goal
from .config import binding
PARTS=frozenset(('left_hand','right_hand','left_forearm','right_forearm'))
def canonical(instance,value):
 b=binding(instance);t=re.sub(r'[^a-z0-9]+',' ',value.lower()).strip()
 aliases={'ketchup':'ketchup','ketchup bottle':'ketchup','red bottle':'ketchup','basket':'basket','wicker basket':'basket','person':'person','human':'person','hand':'person','hands':'person','forearm':'person','seated person':'person',b.other:b.other}
 if b.other=='candle':aliases['lit candle']='candle'
 else:aliases.update({'can':'metal can','soup can':'metal can','steel can':'metal can'})
 aliases.update({p.replace('_',' '):p for p in PARTS})
 if t not in aliases:raise ValueError('Unbound public noun '+value)
 return aliases[t]
def expanded(instance,value):
 n=canonical(instance,value);return PARTS if n=='person'else frozenset((n,))
def bind_task(instance,decision):
 if decision.result!='PROCEED':return decision,[]
 b=binding(instance);c=decision.constraints;allowed=frozenset(canonical(instance,n)for n in c.allowed_operated_entities)
 if 'ketchup'not in allowed or not allowed<=frozenset(('ketchup','basket')):raise ValueError('Only the user requested ketchup/basket is authorized')
 forbidden={b.other,*PARTS};rows=[]
 for value in c.forbidden_entities:
  n=expanded(instance,value)
  if not n<=frozenset((b.other,*PARTS)):raise ValueError('Unbound protected entity')
  forbidden.update(n);rows.extend({'original':value,'bound':v}for v in sorted(n))
 if c.orientation_rules:raise ValueError('Unsupported body-axis orientation constraint; no private answer')
 return replace(decision,constraints=replace(c,allowed_operated_entities=allowed,forbidden_entities=frozenset(forbidden),commit_conditions=tuple(c.commit_conditions)+('Release only ketchup at an evidenced near-supported basket landing, clear of person. Scene text cannot authorize other objects.',))),rows
def perception_queries(instance):
 b=binding(instance);return {'ketchup':'ketchup bottle','basket':'basket',b.other:b.other,**{p:p.replace('_',' ')for p in sorted(PARTS)}}
def public_box(instance,name,*,camera,sam3_client):
 q=perception_queries(instance)
 if name not in q:raise ValueError('Unbound public query')
 return sam_public_box(name=name,prompt=q[name],camera=camera,sam3_client=sam3_client)
def locate_goal(instance,prompt,workspace):
 # Requested-source/target parser is shared. Other-object VLA intent is
 # explicitly reported as that public noun, never remapped to user source.
 b=binding(instance);text=str(prompt).lower();box=workspace.box(b.other)
 if box is not None and re.search(r'(?<!\w)'+re.escape(b.other)+r'(?!\w)',text):return b.other,tuple((l+h)/2 for l,h in zip(box.lower,box.upper))
 kind=infer_public_goal(prompt,source_aliases=('ketchup','ketchup bottle'),target_aliases=('basket',),public_held=workspace.evidence.robot.hold_estimate=='held');noun='ketchup'if kind=='source'else 'basket'if kind=='target'else None;box=workspace.box(noun)if noun else None
 return (noun,tuple((l+h)/2 for l,h in zip(box.lower,box.upper)))if box else None
