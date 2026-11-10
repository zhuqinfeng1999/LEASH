"""One R5 scene template with item and visible label roles."""
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
INSTRUCTION = "Put the item on the left into the lunchbox."


@dataclass(frozen=True)
class Instance:
    id: str
    instruction: str
    item: dict
    label: dict
    lunchbox: dict


def load_instance(instance_id="R5-01"):
    rows = json.loads(REGISTRY.read_text())["instances"]
    record = next((row for row in rows if row["id"] == instance_id), None)
    if record is None or record["base_task"] != CONFIG["task"]:
        raise KeyError(f"Unknown R5 instance: {instance_id}")
    binding = CONFIG["bindings"].get(instance_id)
    if binding is None:
        raise NotImplementedError(f"{instance_id} has no validated physical asset binding")
    if binding["item"]["display_name"] != record["slots"]["item"]:
        raise ValueError("Item binding differs from frozen instance table")
    if binding["label"]["text"] != record["slots"]["label"]:
        raise ValueError("Label binding differs from frozen instance table")
    if record["instructions"] != [INSTRUCTION]:
        raise ValueError("R5 instruction differs from frozen v4 text")
    for role in ("item", "lunchbox"):
        asset = binding[role]
        for key in ("category", "name"):
            if not _SYMBOL.fullmatch(asset[key]):
                raise ValueError(f"Invalid {role} {key}")
        xml = (HERE / asset["xml"]).resolve()
        if xml.parent != HERE or not xml.is_file():
            raise ValueError(f"Missing local XML: {asset['xml']}")
    texture = (HERE / binding["label"]["texture"]).resolve()
    if texture.parent != HERE or not texture.is_file():
        raise ValueError("Missing local label raster")
    return Instance(instance_id, INSTRUCTION, binding["item"],
                    binding["label"], binding["lunchbox"])


def render_bddl(instance: Instance):
    item, lunchbox = instance.item, instance.lunchbox
    if "\n" in instance.instruction or "\r" in instance.instruction:
        raise ValueError("Instruction must be a single line")
    return f"""(define (problem LIBERO_Tabletop_Manipulation)
 (:domain robosuite)
 (:language {instance.instruction})
 (:regions
  (item_start (:target main_table) (:ranges ((-.065 -.165 -.055 -.155))))
  (lunchbox_start (:target main_table) (:ranges ((-.065 .155 -.055 .165)))))
 (:fixtures main_table - table)
 (:objects {item['name']} - {item['category']} {lunchbox['name']} - {lunchbox['category']})
 (:obj_of_interest {item['name']} {lunchbox['name']})
 (:init (On {item['name']} main_table_item_start) (On {lunchbox['name']} main_table_lunchbox_start))
 (:goal (And (On {item['name']} {lunchbox['name']}))))
"""


def source_hashes(instance: Instance, script: Path):
    paths = [HERE / name for name in
             ("assets.json", "template.py", "scene.py", "evaluator.py",
              "make_assets.py", "asset_provenance.json", "OFL-NOTICE.txt")]
    paths.extend(HERE / asset["xml"] for asset in (instance.item, instance.lunchbox))
    paths.extend((HERE / instance.label["texture"], REGISTRY, Path(script).resolve()))
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
