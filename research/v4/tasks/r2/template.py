"""One role-based R1 scene template; asset bindings select the instance."""
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
    source: dict
    target: dict


def load_instance(instance_id=f"{TASK_ID}-01"):
    entries = json.loads(REGISTRY.read_text())["instances"]
    record = next((row for row in entries if row["id"] == instance_id), None)
    if record is None or record["base_task"] != TASK_ID:
        raise KeyError(f"Unknown {TASK_ID} instance: {instance_id}")
    binding = CONFIG["bindings"].get(instance_id)
    if binding is None:
        raise NotImplementedError(f"{instance_id} has no validated physical asset binding yet")
    for role in ("source", "target"):
        asset = binding[role]
        slot = CONFIG["roles"][role]["slot"]
        if asset["display_name"] != record["slots"][slot]:
            raise ValueError(f"{instance_id} {role} binding diverges from the frozen registry")
        for key in ("category", "name"):
            if not _SYMBOL.fullmatch(asset[key]):
                raise ValueError(f"Invalid {role} {key}")
        xml = (HERE / asset["xml"]).resolve()
        if xml.parent != HERE or not xml.is_file():
            raise ValueError(f"Missing or external asset XML for {role}: {asset['xml']}")
    instruction = record["instructions"]
    if len(instruction) != 1:
        raise ValueError(f"{instance_id} must have exactly one instruction")
    return Instance(instance_id, instruction[0], binding["source"], binding["target"])


def render_bddl(instance: Instance):
    """LIBERO syntax generated from roles; one template for every bound instance."""
    source, target = instance.source, instance.target
    # BDDL uses the source text verbatim. A newline would corrupt the syntax.
    if "\n" in instance.instruction or "\r" in instance.instruction:
        raise ValueError("Instruction must be a single line")
    return f"""(define (problem LIBERO_Tabletop_Manipulation)
 (:domain robosuite)
 (:language {instance.instruction})
 (:regions
  (source_start (:target main_table) (:ranges ((-.065 -.165 -.055 -.155))))
  (target_start (:target main_table) (:ranges ((-.065 .155 -.055 .165)))))
 (:fixtures main_table - table)
 (:objects {source['name']} - {source['category']} {target['name']} - {target['category']})
 (:obj_of_interest {source['name']} {target['name']})
 (:init (On {source['name']} main_table_source_start) (On {target['name']} main_table_target_start))
 (:goal (And (On {source['name']} {target['name']}))))
"""


def source_hashes(instance: Instance, script: Path):
    """Hash exact local inputs used by a native owner before it starts."""
    paths = [HERE / name for name in ("assets.json", "template.py", "scene.py", "evaluator.py")]
    paths.extend(HERE / asset["xml"] for asset in (instance.source, instance.target))
    paths.extend((REGISTRY, Path(script).resolve()))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def freeze_inputs(output: Path, hashes: dict[str, str]):
    """Copy every hashed input into this unique owner before simulator startup."""
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
