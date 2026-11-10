"""Complete current union, with every immutable parent obligation retained."""
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = "R301-current-U-REAL-v43-task-lease-main-source-optin-v1"


def validate_packet(actual, expected):
    if not isinstance(expected, dict) or set(actual) != set(expected) or actual != expected:
        raise RuntimeError("complete own R301 main source membership/bytes differ")
    return actual


def admit():
    contract = json.loads((HERE / "contract.json").read_text())
    frozen = json.loads((HERE / "source_freeze.json").read_text())
    if frozen["version"] != VERSION or set(frozen["sources"]) != set(contract["required_members"]):
        raise RuntimeError("whole own R301 main source membership/version differs")
    for parent in contract["parents"]:
        path = Path(parent["path"])
        if sha256(path.read_bytes()).hexdigest() != parent["sha256"]:
            raise RuntimeError("immutable parent freeze changed: " + str(path))
        packet = json.loads(path.read_text())
        packet = packet if parent["field"] is None else packet[parent["field"]]
        for name, expected in packet.items():
            selected_name = parent.get("archive_map", {}).get(name, name)
            absolute = str(Path(selected_name) if Path(selected_name).is_absolute() else ROOT / selected_name)
            if frozen["sources"].get(absolute) != expected:
                raise RuntimeError("complete parent source omitted/changed: " + absolute)
    actual = {}
    for name, expected in frozen["sources"].items():
        path = Path(name)
        if not path.is_file():
            raise RuntimeError("whole own R301 main source missing: " + name)
        actual[name] = sha256(path.read_bytes()).hexdigest()
    validate_packet(actual, frozen["sources"])
    # Each existing full admission retains its own historical/archive semantics.
    from research.v4.candidates.r301_v43_task_lease_consumer_prepare_20260929 import source as consumer
    from research.v4.candidates.r301_current_u_front_ordinary_method_factory_source_prepare_20260929 import binding as factory
    consumer.admit()
    factory.admit()
    return actual


def source_text(name):
    path = Path(name) if Path(name).is_absolute() else ROOT / name
    if str(path) not in admit():
        raise RuntimeError("unadmitted own R301 source: " + str(path))
    return path.read_text()
