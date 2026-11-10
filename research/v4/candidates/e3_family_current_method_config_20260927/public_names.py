"""Public instruction/image noun aliases only. No truth coordinates/dimensions."""
import re
from dataclasses import replace
from research.v4.candidates.e2_laundry_method_20260926.public_intent import infer_public_goal

def bindings(config):
 aliases={x:'vessel'for x in config.source_aliases};aliases.update({x:'target'for x in config.target_aliases});aliases.update({'person':'person','seated person':'person','human':'person','hand':'hand','human hand':'hand','right hand':'hand','left hand':'hand','vessel':'vessel','target':'target'})
 def name(v):
  if not isinstance(v,str):raise ValueError('public entity must be text')
  k=re.sub(r'[^a-z0-9]+',' ',v.casefold()).strip()
  if k not in aliases:raise ValueError('unbound own E3 public name: '+repr(v))
  return aliases[k]
 def bind(decision):
  if decision.result!='PROCEED':return decision,[]
  c=decision.constraints;records=[]
  def names(values,field):
   out=set()
   for v in sorted(values):n=name(v);records.append(dict(field=field,original=v,bound=n));out.add(n)
   return frozenset(out)
  return replace(decision,constraints=replace(c,allowed_operated_entities=names(c.allowed_operated_entities,'allowed_operated_entities'),forbidden_entities=names(c.forbidden_entities,'forbidden_entities'),orientation_rules=tuple(replace(r,operated_entity=name(r.operated_entity))for r in c.orientation_rules))),records
 def locate(prompt,workspace):
  held=workspace.evidence.robot.hold_estimate=='held'
  # The reused small pick/place parser has no 'pour' verb. Bind only a
  # simple explicit held pour-to-named-target clause; no axis/pose inference.
  text=prompt.lower().strip()if isinstance(prompt,str)else''
  pour=bool(held and re.match(r'^pour\b',text)and not re.search(r'\b(?:and|then|do not|never)\b',text)and any(re.search(r'\binto\s+(?:the\s+)?'+re.escape(x)+r'\b',text)for x in config.target_aliases))
  goal='target'if pour else infer_public_goal(prompt,source_aliases=config.source_aliases,target_aliases=config.target_aliases,public_held=held)
  label='vessel'if goal=='source'else'target'if goal=='target'else None;box=workspace.box(label)if label else None
  return(label,tuple((a+b)/2 for a,b in zip(box.lower,box.upper)))if box else None
 return name,bind,locate
