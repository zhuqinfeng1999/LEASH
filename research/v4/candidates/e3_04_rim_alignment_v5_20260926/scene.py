"""Physical scene unchanged; separate offline alignment-control qualification."""
from pathlib import Path
from research.v4.candidates.e3_04_everlift_witness_v4_20260926.scene import Scene as Previous
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTION=Previous.instruction
class Scene(Previous):
 protocol_version='e3-04-supporting-fingers-v3-offline-rim-alignment-v5'
