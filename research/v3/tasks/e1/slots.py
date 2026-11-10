"""E1 slot registry and deterministic BDDL rendering, independent of the evaluator."""
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def load_instance(instance_id):
    registry = json.loads((HERE / "instances.json").read_text())
    spec = registry["instances"][instance_id]
    if not re.fullmatch(r"E1-[0-9]{2}", instance_id):
        raise ValueError("Invalid E1 instance id")
    for slot in ("item", "target"):
        for key in ("object_name", "object_type"):
            if not NAME.fullmatch(spec[slot][key]):
                raise ValueError(f"Invalid {slot}.{key}")
    if not spec["instruction"].startswith("Put ") or any(
        token in spec["instruction"] for token in ("(", ")", "\n")
    ):
        raise ValueError("Instruction is not safe BDDL plain text")
    return spec


def render_bddl(spec):
    text = (HERE / "scene_template.bddl").read_text()
    bindings = {
        "__INSTRUCTION__": spec["instruction"],
        "__ITEM_NAME__": spec["item"]["object_name"],
        "__ITEM_TYPE__": spec["item"]["object_type"],
        "__TARGET_NAME__": spec["target"]["object_name"],
        "__TARGET_TYPE__": spec["target"]["object_type"],
    }
    for token, value in bindings.items():
        text = text.replace(token, value)
    if "__" in text:
        raise ValueError("Unfilled E1 BDDL template slot")
    return text
