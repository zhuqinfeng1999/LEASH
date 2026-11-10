"""Same frozen physical supporting-fingers scene; only offline witness version."""
from pathlib import Path
from research.v4.candidates.e3_04_supporting_fingers_20260926.scene import Scene as Previous
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION=Previous.instruction
class Scene(Previous):
 protocol_version='e3-04-supporting-fingers-v3-offline-everlift-witness-v4'
