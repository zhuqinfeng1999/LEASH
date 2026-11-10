"""Neutral task entities and same fixed human-parts aliases; no hidden truth."""
import re
from dataclasses import replace
from research.v4.accelerated import p1_name_binding as human
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.e2_laundry_method_20260926.public_intent import infer_public_goal
from .config import binding
PARTS=human.PARTS

def canonical(instance,value):
 b=binding(instance);text=re.sub(r'[^a-z0-9]+',' ',value.lower()).strip()
 for noun,aliases in [(b.source,b.source_aliases),(b.target,b.target_aliases)]:
  if text in aliases:return noun
 result=human.canonical_name(value)
 if result in ('knife','plate'):raise ValueError('Unbound other-instance public object '+value)
 return result

def expanded(instance,value):
 x=canonical(instance,value)
 if x=='person':return PARTS
 if x=='hand':return frozenset(('left_hand','right_hand'))
 if x=='forearm_pair':return frozenset(('left_forearm','forearm'))
 return frozenset((x,))

def bind_task(instance,decision):
 if decision.result!='PROCEED':return decision,[]
 b=binding(instance);c=decision.constraints;rows=[];allowed=frozenset(canonical(instance,x)for x in c.allowed_operated_entities)
 if b.source not in allowed or not allowed<=frozenset((b.source,b.target)):raise ValueError('Task source/target scope mismatch')
 forbidden=set(PARTS)
 for value in c.forbidden_entities:
  parts=expanded(instance,value)
  if not parts<=PARTS:raise ValueError('Forbidden P1 entity not public human part')
  forbidden.update(parts);rows.extend({'original':value,'bound':x}for x in sorted(parts))
 rules=[]
 for rule in c.orientation_rules:
  noun=canonical(instance,rule.operated_entity)
  if noun!=b.source:raise ValueError('Orientation rule source mismatch')
  rules.append(replace(rule,operated_entity=noun))
 condition='Never contact the person; keep the requested payload out of both hands and forearms at any height.'
 return replace(decision,constraints=replace(c,allowed_operated_entities=allowed,forbidden_entities=frozenset(forbidden),orientation_rules=tuple(rules),commit_conditions=tuple(c.commit_conditions)+(condition,))),rows

def bind_motion(instance,raw):
 if not isinstance(raw,dict)or not isinstance(raw.get('forbidden_entities'),list):raise ValueError('Malformed motion constraints')
 out=set();rows=[]
 for value in raw['forbidden_entities']:
  for noun in expanded(instance,value):out.add(noun);rows.append({'original':value,'bound':noun})
 return {**raw,'forbidden_entities':sorted(out)},rows

def perception_queries(instance):
 b=binding(instance);return {b.source:b.source,b.target:b.target,'left_hand':'left hand','right_hand':'right hand','left_forearm':'left forearm','forearm':'right forearm'}
def public_box(instance,name,*,camera,sam3_client):
 if name not in perception_queries(instance):raise ValueError('Unbound public query')
 return sam_public_box(name=name,prompt=perception_queries(instance)[name],camera=camera,sam3_client=sam3_client)
def locate_goal(instance,prompt,workspace):
 b=binding(instance);kind=infer_public_goal(prompt,source_aliases=b.source_aliases,target_aliases=b.target_aliases,public_held=workspace.evidence.robot.hold_estimate=='held');noun=b.source if kind=='source'else b.target if kind=='target'else None;box=workspace.box(noun)if noun else None
 if box is None:return None
 return noun,tuple((a+z)/2 for a,z in zip(box.lower,box.upper))
