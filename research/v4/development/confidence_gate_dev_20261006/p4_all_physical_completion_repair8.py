"""Reuse the accepted P401 private actual-grip lifecycle for P402/P403/P404.

Every task retains its own exact physical support and harmful-contact
snapshot. This is scorer coverage, not a new online holding or grasp rule.
"""
from importlib import import_module
from types import SimpleNamespace
from .p4_physical_completion import P401PhysicalCompletionEvaluator
from .p1_physical_completion_repair8 import ExecutedGripperSceneMixin

REVISION = 'p4-private-all-physical-grip-coverage-repair8-20261010'
SNAPSHOTS = {
    'P4-02': 'research.v4.candidates.p4_02_straightener_heatmat_20260926.evaluator',
    'P4-03': 'research.v4.candidates.p4_03_gluegun_20260926.evaluator',
    'P4-04': 'research.v4.candidates.p4_04_soldering_iron_20260926.evaluator',
}


def configure(configured):
    instance = configured.binding.instance
    if instance not in SNAPSHOTS:
        return configured
    parent = configured.Evaluator
    sample = import_module(SNAPSHOTS[instance]).snapshot
    class ActualEvaluator(P401PhysicalCompletionEvaluator, parent):
        physical_grip_parts = None
        physical_revision = REVISION

        def observe(self, scene):
            record = getattr(scene, 'private_last_executed_gripper', None)
            command = (record.get('command') if isinstance(record, dict)
                       and record.get('sequence') == scene.sequence else None)
            return self.observe_row(sample(scene), release_started=scene.last_release_started,
                                    initial_bottom_z=scene.initial_item_bottom_z,
                                    executed_gripper_command=command)
    class ActualScene(ExecutedGripperSceneMixin, configured.Scene):
        pass
    return SimpleNamespace(**{**vars(configured), 'Scene': ActualScene,
                            'Evaluator': ActualEvaluator,
                            'placement_outcome_revision': REVISION})
