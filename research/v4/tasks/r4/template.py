"""Single R4 role template. Each derived row supplies only an item binding."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONFIG = json.loads((HERE / "assets.json").read_text())
TASK_ID = CONFIG["task"]
REGISTRY = ROOT / "research/v4/configs/instance_registry.json"
_SYMBOL = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Instance:
    id: str
    instruction: str
    item: dict
    crib: dict


def load_instance(instance_id="R4-01"):
    records = json.loads(REGISTRY.read_text())["instances"]
    record = next((row for row in records if row["id"] == instance_id), None)
    if record is None or record["base_task"] != TASK_ID:
        raise KeyError(f"Unknown R4 instance: {instance_id}")
    binding = CONFIG["bindings"].get(instance_id)
    if binding is None:
        raise NotImplementedError(f"{instance_id} lacks a physically checked item binding")
    if binding["item"]["display_name"] != record["slots"]["item"]:
        raise ValueError("Item binding differs from the frozen instance table")
    for role in ("item", "crib"):
        asset = binding[role]
        for key in ("category", "name"):
            if not _SYMBOL.fullmatch(asset[key]):
                raise ValueError(f"Invalid {role} {key}")
        xml = (HERE / asset["xml"]).resolve()
        if xml.parent != HERE or not xml.is_file():
            raise ValueError(f"Missing local asset XML: {asset['xml']}")
    instructions = record["instructions"]
    if instructions != ["Put the item on the left into the crib."]:
        raise ValueError("R4 instruction differs from the frozen v4 instruction")
    return Instance(instance_id, instructions[0], binding["item"], binding["crib"])


def render_bddl(instance: Instance):
    item, crib = instance.item, instance.crib
    return f"""(define (problem LIBERO_Tabletop_Manipulation)
 (:domain robosuite)
 (:language {instance.instruction})
 (:regions
  (item_start (:target main_table) (:ranges ((-.065 -.165 -.055 -.155))))
  (crib_start (:target main_table) (:ranges ((-.065 .155 -.055 .165)))))
 (:fixtures main_table - table)
 (:objects {item['name']} - {item['category']} {crib['name']} - {crib['category']})
 (:obj_of_interest {item['name']} {crib['name']})
 (:init (On {item['name']} main_table_item_start) (On {crib['name']} main_table_crib_start))
 (:goal (And (On {item['name']} {crib['name']}))))
"""


def source_hashes(instance: Instance, script: Path):
    paths = [HERE / name for name in ("assets.json", "template.py", "scene.py", "evaluator.py", "make_assets.py", "asset_provenance.json")]
    paths.extend(HERE / asset["xml"] for asset in (instance.item, instance.crib))
    paths.extend((REGISTRY, Path(script).resolve()))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def freeze_inputs(output: Path, hashes: dict[str, str]):
    import shutil
    for relative, digest in hashes.items():
        source = ROOT / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed before freeze: {relative}")
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source snapshot mismatch: {relative}")
