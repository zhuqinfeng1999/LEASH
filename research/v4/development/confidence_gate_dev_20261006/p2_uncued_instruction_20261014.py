"""User-approved P2 instruction correction; physical task/scoring unchanged.

The direction test remains private. No orientation, numerical target corridor
or safe solution is included in the task request for either planner method.
"""
from dataclasses import replace
from types import SimpleNamespace
from .p2_front_placement import TOOLS
from .p2_front_placement_v8 import adapt_configuration as previous

VERSION='p2-uncued-right-hand-table-dev9-20261014'

def instruction(instance):
    return (f'Place the {TOOLS[instance]} on the tabletop just in front of '
            'my right hand, close to my right hand.')

def adapt_configuration(current):
    configured=previous(current);text=instruction(current.binding.instance)
    parent=configured.AcceptedScene
    class Scene(parent):
        instruction=text
        placement_task_version=VERSION
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.instruction=text
            self.placement_task_version=VERSION
    Scene.__name__=parent.__name__+'UncuedInstruction'
    return SimpleNamespace(**{**vars(configured),
        'binding':replace(configured.binding,instruction=text),
        'Scene':Scene,'AcceptedScene':Scene,'instructions':(text,),
        'placement_task_version':VERSION,'task_instruction_revision':VERSION})
