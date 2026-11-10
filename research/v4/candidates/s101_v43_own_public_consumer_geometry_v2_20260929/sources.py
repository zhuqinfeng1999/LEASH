"""Complete immutable parent source admission before v2 selection."""
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = "s101-v43-own-public-geometry-source-v2"
PARENT = "research/v4/candidates/s101_v43_own_public_consumer_20260929/source_freeze.json"
PARENT_SHA = "0ec44a48a60ea7d3ce58814a16c9ad9c44f6f2c82615df7e4997cff5cab1fc82"
RUNTIME_FILES = ("__init__.py", "runtime.py", "sources.py", "check_mock.py", "freeze_sources.py")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def source_union():
    path = ROOT / PARENT
    if digest(path) != PARENT_SHA:
        raise RuntimeError("immutable 2965-source parent freeze file changed")
    result = dict(json.loads(path.read_text())["sha256"])
    for key, expected in result.items():
        p = Path(key) if Path(key).is_absolute() else ROOT / key
        if digest(p) != expected:
            raise RuntimeError("immutable parent source changed: " + key)
    result[PARENT] = PARENT_SHA
    for name in RUNTIME_FILES:
        p = HERE / name
        result[str(p.relative_to(ROOT))] = digest(p)
    return dict(sorted(result.items()))


def source_hashes(expected=None):
    frozen = json.loads((HERE / "source_freeze.json").read_text())
    if (frozen["version"] != VERSION or frozen["launch_authorized"] is not False
            or frozen["actual_main_consumed"] is not False):
        raise RuntimeError("v2 source version/launch scope changed")
    current = source_union()
    if current != (frozen["sha256"] if expected is None else expected):
        raise RuntimeError("complete own geometry v2 source admission differs")
    return current


def make_consumer(*args, **kwargs):
    source_hashes()
    from .runtime import S101OwnPublicConsumer
    return S101OwnPublicConsumer(*args, **kwargs)
