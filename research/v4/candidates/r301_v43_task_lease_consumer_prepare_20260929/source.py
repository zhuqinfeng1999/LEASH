"""Whole new Consumer source plus the independently qualified provider packet."""
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def admit():
    contract = json.loads((HERE / "contract.json").read_text())
    expected = json.loads((HERE / "source_freeze.json").read_text())
    if set(contract["required_members"]) - set(expected):
        raise RuntimeError("complete R301 Consumer source membership missing")
    for name, wanted in expected.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != wanted:
            raise RuntimeError(f"whole R301 Consumer source changed: {name}")
    from research.v4.candidates.r301_v43_same_packet_public_provider_py311_contract_v3_prepare_20260929 import source as qualified
    qualified.admit()  # Complete 963 + historical914/archive obligations unchanged.
    return expected


def source_text(name):
    if name not in admit():
        raise RuntimeError("unadmitted R301 public source")
    return (ROOT / name).read_text()
