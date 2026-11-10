"""Unchanged v5 physical scene; live-target OFFLINE script is separate."""
from pathlib import Path
from research.v4.candidates.e3_04_rim_alignment_v5_20260926.scene import Scene as Parent
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
class Scene(Parent):
    protocol_version = 'e3-04-supporting-fingers-offline-live-rim-transport-v6'
