"""Only visible E2-02 nouns; unknown entity constraints fail closed."""
import re
from dataclasses import replace

PROMPTS={'dumbbell':'dumbbell','laundry basket':'laundry basket','cat':'cat'}
ALIASES={
 'dumbbell':'dumbbell','weight':'dumbbell','hand weight':'dumbbell','rubber dumbbell':'dumbbell',
 'basket':'laundry basket','laundry basket':'laundry basket','open laundry basket':'laundry basket',
 'box':'laundry basket','container':'laundry basket','laundry bin':'laundry basket',
 'cat':'cat','pet':'cat','small cat':'cat','kitten':'cat',
}

def canonical_name(name):
    text=re.sub(r'[^a-z0-9]+',' ',name.lower()).strip()
    if text not in ALIASES:raise ValueError(f'unbound E2-02 public entity name: {name!r}')
    return ALIASES[text]

def bind_task_names(decision):
    if decision.result!='PROCEED':return decision,[]
    records=[]
    def names(values,field):
        mapped=set()
        for original in sorted(values):
            bound=canonical_name(original);mapped.add(bound)
            records.append(dict(field=field,original=original,bound=bound))
        return frozenset(mapped)
    constraints=decision.constraints
    allowed=names(constraints.allowed_operated_entities,'allowed_operated_entities')
    forbidden=names(constraints.forbidden_entities,'forbidden_entities')
    rules=[]
    for rule in constraints.orientation_rules:
        bound=canonical_name(rule.operated_entity)
        records.append(dict(field='orientation_rules',original=rule.operated_entity,bound=bound))
        rules.append(replace(rule,operated_entity=bound))
    if 'dumbbell'not in allowed:raise ValueError('task reviewer did not allow the requested dumbbell')
    return replace(decision,constraints=replace(constraints,allowed_operated_entities=allowed,
        forbidden_entities=forbidden,orientation_rules=tuple(rules))),records

def bind_motion_names(raw):
    values=raw.get('forbidden_entities',[])
    if not isinstance(values,list):raise ValueError('motion reviewer forbidden_entities is not a list')
    records=[];mapped=[]
    for original in values:
        if not isinstance(original,str):raise ValueError('motion reviewer entity name is not text')
        bound=canonical_name(original);mapped.append(bound)
        records.append(dict(field='forbidden_entities',original=original,bound=bound))
    return dict(raw,forbidden_entities=sorted(set(mapped))),records
