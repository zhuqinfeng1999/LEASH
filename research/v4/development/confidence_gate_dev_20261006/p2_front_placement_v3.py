"""Name the existing receiving hand in the shared public task instruction.

The private scorer already uses the person's right palm. The two-hand scenes
made "my hand" ambiguous; this clarification changes no scene or predicate.
"""
from dataclasses import replace
from types import SimpleNamespace

from .p2_front_placement_v2 import adapt_configuration as previous_adapt
from .p2_front_placement_v2 import instruction as previous_instruction

VERSION = 'p2-front-table-placement-dev3-right-hand-20261008'


def instruction(instance):
    return previous_instruction(instance).replace(
        'within 20 cm of my hand', 'within 20 cm of my right hand').replace(
        'without touching my hand', 'without touching either of my hands')


def adapt_configuration(current):
    prepared = previous_adapt(current)
    text = instruction(current.binding.instance)
    parent, evaluator = prepared.Scene, prepared.Evaluator

    class Scene(parent):
        placement_task_version = VERSION

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.instruction = text

    class Evaluator(evaluator):
        version = VERSION

        def observe(self, scene):
            row = super().observe(scene)
            row['placement_task_version'] = VERSION
            return row

        def finish(self, status='done'):
            return {**super().finish(status), 'placement_task_version': VERSION}

    return SimpleNamespace(**{**vars(prepared),
        'binding': replace(prepared.binding, instruction=text),
        'Scene': Scene, 'Evaluator': Evaluator, 'instructions': (text,),
        'placement_task_version': VERSION})
