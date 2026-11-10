"""E3-01's original kettle/mug, with the approved whole-person rest pose only."""
from pathlib import Path
from research.v4.candidates.e3_relaxed_initial_person_20260926.scene import RelaxedKettleFacade
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
class Scene(RelaxedKettleFacade):
    pass
