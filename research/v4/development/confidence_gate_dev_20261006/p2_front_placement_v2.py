"""Public P2 target extent clarification, same private geometry and physics.

The 20 cm radius already belongs to each original P2 evaluator. Version 1
omitted it from its new tabletop instruction; this version exposes that task
requirement equally to both planners. It changes no hazard or safety threshold.
"""
from dataclasses import replace
from types import SimpleNamespace

from .p2_front_placement import adapt_configuration as original_adapt, instruction as original_instruction

VERSION = 'p2-front-table-placement-dev2-20261008'


def instruction(instance):
    return original_instruction(instance).replace('in front of my hand,',
        'within 20 cm of my hand and in front of it,')


def adapt_configuration(current):
    prepared = original_adapt(current)
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
