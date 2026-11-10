"""P1-03 ONLY physical bridge-region annotation fix, no control/asset change.

The object-local grasp site is a physical between-finger region reference,
not an EEF standoff. The two actual bridge capsules lie at z=0/radius6mm;
z=28mm is outside them. Preserve x/y, original assist tolerance and geometry.
"""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
VERSION='P103-physical-bridge-grasp-center-z0-dev1-20261014'
OLD=(-.036,0.,.028)
NEW=(-.036,0.,0.)

def corrected_bindings(bindings):
    result=[];count=0
    for binding in bindings:
        if binding.name=='blunt_tool_1':
            if tuple(binding.grasp_local)!=OLD:raise ValueError('P103 original grasp annotation differs from pinned known source')
            binding=replace(binding,grasp_local=NEW);count+=1
        result.append(binding)
    if count!=1:raise ValueError('Expected exactlyone P103 scissors physical binding')
    return tuple(result)

def configure(configured):
    if configured.binding.instance!='P1-03':return configured
    from research.v4.candidates.p1_03_scissors_case_20260926 import scene as physical
    parent=configured.Scene
    bindings=corrected_bindings(physical.BINDINGS)
    class BridgeCenterScene(parent):
        physical_grasp_annotation_revision=VERSION
        def __init__(self,*args,**kwargs):
            with patch.object(physical,'BINDINGS',bindings):super().__init__(*args,**kwargs)
    return SimpleNamespace(**(vars(configured)|dict(Scene=BridgeCenterScene,physical_grasp_annotation_revision=VERSION)))
