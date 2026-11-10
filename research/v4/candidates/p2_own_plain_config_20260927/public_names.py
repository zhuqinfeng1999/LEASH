"""Public user vocabulary only; no authored handle/tip/grasp coordinates."""
import re
from dataclasses import replace
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.e2_laundry_method_20260926.public_intent import infer_public_goal
from .config import binding
PARTS=frozenset(('left_hand','right_hand','left_forearm','right_forearm'))
def canonical(instance,value):
 b=binding(instance);t=re.sub(r'[^a-z0-9]+',' ',value.lower()).strip()
 if t in b.source_aliases:return b.source
 if t in ('person','human','seated person'):return 'person'
 if t in ('hand','hands'):return 'hand'
 if t in ('forearm','forearms'):return 'forearms'
 for p in PARTS:
  if t==p.replace('_',' '):return p
 raise ValueError('Unbound public entity '+value)
def expanded(instance,value):
 n=canonical(instance,value)
 return PARTS if n=='person' else frozenset(('left_hand','right_hand'))if n=='hand'else frozenset(('left_forearm','right_forearm'))if n=='forearms'else frozenset((n,))
def bind_task(instance,decision):
 if decision.result!='PROCEED':return decision,[]
 b=binding(instance);c=decision.constraints;allowed=frozenset(canonical(instance,x)for x in c.allowed_operated_entities)
 if allowed!=frozenset((b.source,)):raise ValueError('Only requested tool may be operated')
 out=set(PARTS);rows=[]
 for value in c.forbidden_entities:
  p=expanded(instance,value)
  if not p<=PARTS:raise ValueError('Protected entity must be public human part')
  out.update(p);rows.extend({'original':value,'bound':n}for n in sorted(p))
 rules=[]
 for r in c.orientation_rules:
  if canonical(instance,r.operated_entity)!=b.source:raise ValueError('Wrong orientation entity')
  rules.append(replace(r,operated_entity=b.source))
 condition='Release only onto clear stable tabletop beside the hand; handle toward recipient, all sharp tips away. Missing public direction evidence means BLOCK; AABB/EEF are not body-axis evidence.'
 return replace(decision,constraints=replace(c,allowed_operated_entities=allowed,forbidden_entities=frozenset(out),orientation_rules=tuple(rules),commit_conditions=tuple(c.commit_conditions)+(condition,))),rows
def bind_motion(instance,raw):
 if not isinstance(raw,dict)or not isinstance(raw.get('forbidden_entities'),list):raise ValueError('Malformed motion constraints')
 out=set();rows=[]
 for value in raw['forbidden_entities']:
  for n in expanded(instance,value):out.add(n);rows.append({'original':value,'bound':n})
 return {**raw,'forbidden_entities':sorted(out)},rows
def perception_queries(instance):
 b=binding(instance);return {b.source:b.source,**{p:p.replace('_',' ')for p in sorted(PARTS)}}
def public_box(instance,name,*,camera,sam3_client):
 q=perception_queries(instance)
 if name not in q:raise ValueError('Unbound query')
 return sam_public_box(name=name,prompt=q[name],camera=camera,sam3_client=sam3_client)
def locate_goal(instance,prompt,workspace):
 b=binding(instance);kind=infer_public_goal(prompt,source_aliases=b.source_aliases,target_aliases=b.target_aliases,public_held=workspace.evidence.robot.hold_estimate=='held');noun=b.source if kind=='source'else b.target if kind=='target'else None;box=workspace.box(noun)if noun else None
 return (noun,tuple((a+z)/2 for a,z in zip(box.lower,box.upper)))if box else None
