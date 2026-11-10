"""Instruction noun aliases, existing SAM projection; no private geometry."""
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.e3_family_current_method_config_20260927.public_names import bindings
from .config import CONFIG, INSTANCE

canonical_name, _task_names, _locate = bindings(CONFIG)
PROMPTS = {'vessel': CONFIG.source_aliases[0], 'target': CONFIG.target_aliases[0],
           'person': 'seated person', 'hand': 'human hand'}

def require_instance(instance):
    if instance != INSTANCE:
        raise ValueError('exact own E305 public binding required')

def perception_queries(instance):
    require_instance(instance)
    return dict(PROMPTS)

def public_box(instance, name, *, camera, sam3_client):
    require_instance(instance)
    noun = canonical_name(name)
    return sam_public_box(name=noun, prompt=PROMPTS[noun], camera=camera,
                          sam3_client=sam3_client)

def bind_task_names(decision):
    result, records = _task_names(decision)
    if result.result == 'PROCEED' and 'vessel' not in result.constraints.allowed_operated_entities:
        raise ValueError('task reviewer did not allow the requested vessel')
    return result, records

def bind_motion_names(raw):
    values = raw.get('forbidden_entities', [])
    if not isinstance(values, list):
        raise ValueError('motion reviewer forbidden_entities is not a list')
    records = [{'field': 'forbidden_entities', 'original': v,
                'bound': canonical_name(v)} for v in values]
    return {**raw, 'forbidden_entities': sorted({r['bound'] for r in records})}, records

def locate_goal(instance, prompt, workspace):
    require_instance(instance)
    return _locate(prompt, workspace)
