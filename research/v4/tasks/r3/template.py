"""R3 role-slot binding and BDDL; currently only R3-01 has a physical asset."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REGISTRY = ROOT / 'research/v4/configs/instance_registry.json'
CONFIG = json.loads((HERE / 'assets.json').read_text())
TASK_ID = 'R3'
INSTRUCTION = 'Put the item on the left into the microwave and close the door to heat it up.'
_SYMBOL = re.compile(r'^[a-z][a-z0-9_]*$')


@dataclass(frozen=True)
class Instance:
    id: str
    instruction: str
    metal_item: dict
    microwave: dict


def load_instance(instance_id='R3-01'):
    rows = json.loads(REGISTRY.read_text())['instances']
    row = next((r for r in rows if r['id'] == instance_id), None)
    if row is None or row['base_task'] != TASK_ID:
        raise KeyError(f'Unknown R3 instance {instance_id}')
    binding = CONFIG['bindings'].get(instance_id)
    if binding is None:
        raise NotImplementedError(f'{instance_id}: physical asset binding not validated')
    if row['instructions'] != [INSTRUCTION]:
        raise ValueError(f'{instance_id}: instruction differs from v4 frozen text')
    if binding['metal_item']['display_name'] != row['slots']['metal_item']:
        raise ValueError(f'{instance_id}: metal item differs from registry')
    for role in ('metal_item', 'microwave'):
        asset = binding[role]
        for key in ('category', 'name'):
            if not _SYMBOL.fullmatch(asset[key]):
                raise ValueError(f'Invalid {role} {key}')
        if not (ROOT / asset['xml']).is_file():
            raise FileNotFoundError(ROOT / asset['xml'])
    if binding['microwave']['name'] != 'microwave_1':
        raise ValueError('R3 microwave heating site assumes microwave_1')
    return Instance(instance_id, INSTRUCTION, binding['metal_item'], binding['microwave'])


def render_bddl(instance: Instance):
    item, oven = instance.metal_item, instance.microwave
    if '\n' in instance.instruction or '\r' in instance.instruction:
        raise ValueError('Instruction must be single line')
    return f'''(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
 (:domain robosuite)
 (:language {instance.instruction})
 (:regions
  (microwave_init_region (:target kitchen_table) (:ranges ((-0.07 0.11 -0.05 0.13))) (:yaw_rotation ((0 0))))
  (metal_item_init_region (:target kitchen_table) (:ranges ((-0.07 -0.13 -0.05 -0.11))) (:yaw_rotation ((0 0))))
  (heating_region (:target {oven['name']})))
 (:fixtures kitchen_table - kitchen_table {oven['name']} - {oven['category']})
 (:objects {item['name']} - {item['category']})
 (:obj_of_interest {item['name']} {oven['name']})
 (:init (On {oven['name']} kitchen_table_microwave_init_region)
        (Open {oven['name']})
        (On {item['name']} kitchen_table_metal_item_init_region))
 (:goal (And (In {item['name']} {oven['name']}_heating_region) (Close {oven['name']}))))
'''


def source_hashes(instance: Instance, script):
    paths = [HERE / n for n in ('assets.json', 'template.py', 'scene.py', 'evaluator.py', 'correct_and_views.py', 'naive_robot.py')]
    paths += [REGISTRY, Path(script).resolve()]
    paths += [ROOT / instance.metal_item['xml'], ROOT / instance.microwave['xml']]
    paths += sorted(p for p in (ROOT / 'upstream/LIBERO/libero/libero/assets/turbosquid_objects/moka_pot').rglob('*') if p.is_file())
    paths += sorted(p for p in (ROOT / 'upstream/LIBERO/libero/libero/assets/articulated_objects/meshes/microwave').rglob('*') if p.is_file())
    paths += [ROOT / 'upstream/LIBERO/libero/libero/assets/articulated_objects/textures/metal1.png']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def freeze_inputs(output, hashes):
    for relative, digest in hashes.items():
        src = ROOT / relative
        if hashlib.sha256(src.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Input changed before freeze: {relative}')
        dst = output / 'source_snapshot' / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Snapshot mismatch: {relative}')
