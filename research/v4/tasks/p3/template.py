"""Frozen P3-01 role binding and task-local source inventory."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REGISTRY = ROOT / "research/v4/configs/instance_registry.json"
INSTRUCTION = "Put the cup on the coaster on the right."


@dataclass(frozen=True)
class Instance:
    id: str
    instruction: str
    vessel: str
    protected: str
    target: str


def load_instance(instance_id="P3-01"):
    rows = json.loads(REGISTRY.read_text())["instances"]
    row = next((item for item in rows if item["id"] == instance_id), None)
    if row is None or row["base_task"] != "P3":
        raise KeyError(instance_id)
    if instance_id != "P3-01":
        raise NotImplementedError("Derived P3 roles require their own checked assets")
    if row["instructions"] != [INSTRUCTION]:
        raise ValueError("P3-01 instruction differs from frozen v4 registry")
    if row["slots"] != {"vessel": "一杯水", "protected": "笔记本电脑", "target": "杯垫"}:
        raise ValueError("P3-01 role binding differs from frozen v4 registry")
    return Instance(instance_id, INSTRUCTION, "cup65", "open_laptop", "coaster")


def scene_bddl(instance=None):
    instance = instance or load_instance()
    text = (HERE / "scene.bddl").read_text()
    if f"(:language {instance.instruction})" not in text:
        raise ValueError("Scene BDDL instruction differs from frozen v4 registry")
    return text


def source_paths(script):
    """Pin exactly the P3 runtime source and referenced assets, not old rasters."""
    paths = [HERE / name for name in (
        "scene.py", "evaluator.py", "template.py", "preflight.py",
        "regression.py", "build_key_atlas.py", "build_screen.py",
        "laptop_template.xml", "scene.bddl", "coaster.xml",
        "assets.json", "LICENSE",
    )]
    paths += [REGISTRY, Path(script).resolve()]
    paths += [p for p in (HERE / "assets").iterdir() if p.is_file()]
    v1 = ROOT / "research/v2/assets/cup-laptop-v1"
    paths += [v1 / name for name in (
        "keycap.obj", "laptop_base.obj", "laptop_lid.obj",
        "screen_face.obj", "LICENSE",
    )]
    v2 = ROOT / "research/v2/assets/cup-laptop-v2"
    paths += [v2 / name for name in ("key_legends.obj", "LICENSE")]
    clean_cup_xml = HERE / "assets/liquid_proxy_cup_clean.xml"
    paths += [(clean_cup_xml.parent / mesh.get("file")).resolve()
              for mesh in ET.parse(clean_cup_xml).find("asset").findall("mesh")]
    return sorted(set(paths))


def source_hashes(script):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths(script)}


def freeze_inputs(output, hashes):
    output = Path(output)
    for relative, expected in hashes.items():
        source = ROOT / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed before freeze: {relative}")
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source snapshot mismatch: {relative}")
