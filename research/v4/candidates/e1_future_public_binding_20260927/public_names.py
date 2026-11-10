"""Task-local neutral noun aliases over existing public perception helpers."""
from dataclasses import replace
import re
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.accelerated.e1_gate_boundary import requires_descent_review
from research.v4.candidates.e2_laundry_method_20260926.public_intent import infer_public_goal
from .config import binding


def normalized(text):
    if not isinstance(text, str):
        raise ValueError('entity name must be text')
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def canonical(instance, name):
    b = binding(instance)
    text = normalized(name)
    groups = [(b.source, b.source_aliases), (b.target, b.target_aliases),
              ('person', ('person', 'seated person', 'human', 'human body')),
              ('hand', ('hand', 'human hand', 'left hand', 'right hand', 'person s hand'))]
    matches = [noun for noun, aliases in groups if text in {normalized(x) for x in aliases}]
    if len(matches) != 1:
        raise ValueError('unbound/ambiguous public E1 name: ' + repr(name))
    return matches[0]


def bind_task(instance, decision):
    if decision.result != 'PROCEED':
        return decision, []
    b = binding(instance); records = []
    def names(values, field):
        out = set()
        for original in sorted(values):
            name = canonical(instance, original)
            records.append({'field': field, 'original': original, 'bound': name})
            out.add(name)
        return frozenset(out)
    c = decision.constraints
    allowed = names(c.allowed_operated_entities, 'allowed_operated_entities')
    forbidden = names(c.forbidden_entities, 'forbidden_entities')
    orientation = []
    for rule in c.orientation_rules:
        name = canonical(instance, rule.operated_entity)
        records.append({'field': 'orientation_rules', 'original': rule.operated_entity, 'bound': name})
        orientation.append(replace(rule, operated_entity=name))
    if b.source not in allowed:
        raise ValueError('task review did not permit the user-named source')
    return replace(decision, constraints=replace(c, allowed_operated_entities=allowed,
        forbidden_entities=forbidden, orientation_rules=tuple(orientation))), records


def bind_motion(instance, raw):
    if not isinstance(raw, dict) or not isinstance(raw.get('forbidden_entities'), list):
        raise ValueError('motion reviewer forbidden list required')
    records = []; mapped = []
    for original in raw['forbidden_entities']:
        name = canonical(instance, original)
        records.append({'field': 'forbidden_entities', 'original': original, 'bound': name})
        mapped.append(name)
    return {**raw, 'forbidden_entities': mapped}, records


def perception_queries(instance):
    b = binding(instance)
    # Neutral user-visible descriptions; no truth positions or risk labels.
    return {b.source: b.source, b.target: b.target,
            'person': 'seated person', 'hand': 'human hand'}


def public_box(instance, name, *, camera, sam3_client):
    noun = canonical(instance, name)
    prompt = perception_queries(instance)[noun]
    return sam_public_box(name=noun, prompt=prompt, camera=camera, sam3_client=sam3_client)


def locate_goal(instance, prompt, workspace):
    b = binding(instance)
    kind = infer_public_goal(prompt, source_aliases=b.source_aliases,
        target_aliases=b.target_aliases,
        public_held=workspace.evidence.robot.hold_estimate == 'held')
    noun = b.source if kind == 'source' else b.target if kind == 'target' else None
    box = workspace.box(noun) if noun else None
    if box is None or box.source != 'public_rgbd_perception':
        return None
    return noun, tuple((lo + hi) / 2 for lo, hi in zip(box.lower, box.upper))


def phase_requires_review(*, public_held, primitive_name, current_eef_xyz, proposed_eef_xyz):
    # Exact existing public phase selector, never hand/evaluator-trigger truth.
    return requires_descent_review(held_public=public_held, primitive_name=primitive_name,
        current_eef_xyz=current_eef_xyz, proposed_eef_xyz=proposed_eef_xyz)
