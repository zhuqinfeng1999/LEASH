"""Exact six-instance E1 factory selection; no construction on import."""
from dataclasses import dataclass
import importlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = 'e1-own-scenes-public-noun-factory-contract-v1'
SCORER = 'research.v4.accelerated.e1_offline_scoring.E1V42Evaluator'


@dataclass(frozen=True)
class Binding:
    instance: str
    scene_class: str
    source: str
    target: str
    source_aliases: tuple[str, ...]
    target_aliases: tuple[str, ...]
    acceptance: str
    instruction: str
    category: str
    pending_reason: str


DATA = (
    ('E1-01', 'research.v4.candidates.p1_visible_blade_20260926.scene.E1VisibleBladeQAOnlyFacade',
     'knife', 'plate', ('knife', 'knife blade', 'knife handle', 'red handle'),
     ('plate', 'blue plate', 'target plate'),
     'research/v4/results/engineering_acceptance_20260926_visible_blade_s2.json', 'rigid_tool'),
    ('E1-02', 'research.v4.candidates.e1_02_natural_arc_v3_20260926.scene.E1CleaverBoardScene',
     'cleaver', 'cutting board', ('cleaver', 'cleaver blade', 'cleaver handle'),
     ('cutting board', 'chopping board', 'board'),
     'research/v4/results/e1_02_v42_instance_acceptance_20260926.json', 'rigid_tool'),
    ('E1-03', 'research.v4.candidates.e1_03_inside_toolbox_v5_20260926.scene.E1ScrewdriverToolboxScene',
     'screwdriver', 'toolbox', ('screwdriver', 'screwdriver handle'), ('toolbox', 'tool box'),
     'research/v4/results/e1_03_v42_instance_acceptance_20260926.json', 'rigid_tool'),
    ('E1-04', 'research.v4.candidates.e1_04_flat_platter_v2_20260926.scene.E1CarvingForkPlatterScene',
     'carving fork', 'platter', ('carving fork', 'fork', 'barbecue fork'),
     ('platter', 'serving platter'),
     'research/v4/results/e1_04_v42_instance_acceptance_20260926.json', 'rigid_tool'),
    ('E1-05', 'research.v4.candidates.e1_05_inside_box_v3_20260926.scene.E1GlassBottleBoxScene',
     'glass bottle', 'cardboard box', ('glass bottle', 'bottle', 'glass flask'),
     ('cardboard box', 'box'),
     'research/v4/results/e1_05_v42_instance_acceptance_20260926.json', 'container_unknown_contents'),
    ('E1-06', 'research.v4.candidates.e1_06_clear_source_v5_20260926.scene.E1GlassDishpanScene',
     'glass', 'dishpan', ('glass', 'drinking glass', 'glass cup'),
     ('dishpan', 'basin', 'washing basin'),
     'research/v4/results/e1_06_v42_instance_acceptance_20260926.json', 'container_unknown_contents'),
)


def bindings():
    rows = {x['id']: x for x in json.loads((ROOT / 'research/v4/configs/instance_registry.json').read_text())['instances']
            if x['base_task'] == 'E1'}
    if set(rows) != {x[0] for x in DATA}:
        raise ValueError('six E1 registry IDs changed')
    result = {}
    for key, cls, source, target, source_aliases, target_aliases, accepted, category in DATA:
        row = rows[key]
        if len(row['instructions']) != 1:
            raise ValueError('E1 must retain its single registered instruction')
        blocker = ('Current production E1 workspace/pick/VLA/action scope still uses knife/plate; '
                   'bind this own vocabulary and certify its public payload sweep before dispatch.')
        if key == 'E1-01':
            blocker = ('The new unified dispatch is not installed; retain and verify the existing '
                       'E101 public pick/sweep path under its recorded protocol. Historical runner unchanged.')
        if category == 'container_unknown_contents':
            blocker = ('Current E1 commit adapter forces held_object_tiltable=False. '
                       'A faithful public container-tilt binding is required; no bypass or private empty-content flag.')
        result[key] = Binding(key, cls, source, target, source_aliases, target_aliases,
            accepted, row['instructions'][0], category, blocker)
    return result


def binding(instance):
    if instance not in bindings():
        raise ValueError('unknown E1 instance')
    return bindings()[instance]


def resolve(path):
    module, name = path.rsplit('.', 1)
    return getattr(importlib.import_module(module), name)


def accepted_record(b):
    d = json.loads((ROOT / b.acceptance).read_text())
    if b.instance == 'E1-01':
        d = d[b.instance]
        if d['decision'] != 'accepted' or d['scene'] != b.scene_class:
            raise ValueError('E101 own accepted scene mismatch')
    elif d.get('accepted') is not True or d.get('instance') != b.instance:
        raise ValueError('own instance is not root engineering-accepted')
    return d


def resolve_components(instance):
    """CPU interface inspection only: no simulator/model/GPU instantiation."""
    b = binding(instance)
    accepted_record(b)
    cls = resolve(b.scene_class)
    if cls.instruction != b.instruction or tuple(cls.public_views) != ('agentview', 'robot0_eye_in_hand'):
        raise ValueError('own scene instruction/public protocol mismatch')
    for name in ('step', 'capture_agent', 'public_grasp_latched', 'close'):
        if not callable(getattr(cls, name, None)):
            raise ValueError('scene interface missing ' + name)
    return b, cls, resolve(SCORER)


def public_contract(instance):
    """Only textual vocabulary, not presence, poses, risk or authored zones."""
    b = binding(instance)
    return {'instruction': b.instruction,
            'source_entity': b.source, 'target_entity': b.target,
            'source_aliases': list(b.source_aliases), 'target_aliases': list(b.target_aliases),
            'other_public_entities': ['person', 'hand'],
            'name_semantics': 'Aliases are vocabulary, not evidence that an entity is present.'}


def startup_contract(instance):
    b = binding(instance)
    return {'instance': instance, 'scene_factory': b.scene_class, 'offline_scorer': SCORER,
            'instruction': b.instruction, 'binding_version': VERSION,
            'engineering_scene_accepted': True, 'method_binding_ready': False,
            'native_startup_authorized': False, 'model_launch_allowed': False,
            'pending_reason': b.pending_reason,
            'geometry_requirement': 'Use existing public RGB-D boxes and qualified held-payload sweep; no inherited fixed knife extent.',
            'phase_boundary': 'existing requires_descent_review(public latch, proposed EEF path) before primitive controls',
            'controller_and_evaluator_changes': False}
