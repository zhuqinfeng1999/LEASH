"""Small public-language goal parser; no perception, coordinates or task truth.

Returns source/target/None. Vocabulary is supplied by the instance wrapper.
This is intentionally conservative, not a general-language safety oracle.
"""
import re

ACTION=re.compile(r'\b(pick(?: up)?|grasp|lift|put|place|drop|release|deposit|transport|carry)\b')
PLACEMENT={'put','place','drop','release','deposit','transport','carry'}
RELATION=re.compile(r'\b(?:on|to|into|in|inside|onto|beside|near|next to|left of|right of|that|and)\b')
DESTINATION=re.compile(r'\b(?:to|into|in|inside|onto|on)\b')

def _mentions(text,vocabulary):
    return any(re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)',text)for word in vocabulary)

def infer_public_goal(prompt,*,source_aliases,target_aliases,public_held=False):
    if not isinstance(prompt,str):return None
    text=prompt.lower()
    # Text after an explicit prohibition/stop is not an executed action goal.
    text=re.split(r"\b(?:do not|don't|never|stop before|without)\b",text,maxsplit=1)[0]
    actions=list(ACTION.finditer(text))
    if not actions:return None
    source_selected=False;goals=[]
    for index,action in enumerate(actions):
        end=actions[index+1].start()if index+1<len(actions)else len(text)
        clause=re.split(r'[.;]',text[action.end():end],maxsplit=1)[0]
        direct=RELATION.split(clause,maxsplit=1)[0]
        named_source=_mentions(direct,source_aliases)
        # A later "lift it" can refer to this prompt's explicit source pick;
        # held public state similarly supports "place it into the basket".
        pronoun=bool(re.search(r'\bit\b',direct))and(source_selected or public_held)
        if not(named_source or pronoun):
            # A bare "grasp and lift the source" has no contrary direct noun.
            if not direct.strip(' ,:')and action.group()not in PLACEMENT:continue
            return None
        if action.group()in PLACEMENT:
            target_clause=DESTINATION.split(clause,maxsplit=1)
            if len(target_clause)!=2 or not _mentions(target_clause[1],target_aliases):return None
            goals.append('target')
        else:
            source_selected=True;goals.append('source')
    return goals[-1]if goals else None
