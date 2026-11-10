"""Continuous-range-derived whole-crib Y slot; own carrier otherwise unchanged."""
import copy,json
from types import FunctionType
from unittest.mock import patch
from .derive import HERE,ROOT
from ..r401_normal_crib_legacy_slot_v2 import scene as old
GEOMETRY=json.loads((HERE/'geometry_range.json').read_text())
Y=GEOMETRY['derivation']['new_nominal_crib_y_m']
ROLE_NOMINAL={'item':[-.06,-.16],'crib':[-.48,Y]}
def render_bddl(instance):
 source=old.original.render_bddl(instance)
 before='(crib_start (:target main_table) (:ranges ((-.065 .155 -.055 .165))))'
 after=f'(crib_start (:target main_table) (:ranges ((-.485 {Y-.005:.12g} -.475 {Y+.005:.12g}))))'
 if source.count(before)!=1:raise RuntimeError('Own original BDDL slot boundary changed')
 return source.replace(before,after)
INIT_NS={**old.SELECTED_INIT.__globals__,'ROLE_NOMINAL':ROLE_NOMINAL,'render_bddl':render_bddl}
SELECTED_INIT=FunctionType(old.SELECTED_INIT.__code__,INIT_NS,name='__init__',argdefs=old.SELECTED_INIT.__defaults__)
class Scene(old.Scene):
 def __init__(self,*a,**kw):
  with patch.object(old,'SELECTED_INIT',SELECTED_INIT),patch.object(old,'ROLE_NOMINAL',ROLE_NOMINAL):super().__init__(*a,**kw)
  self.layout_version='R401-one-continuous-domain-whole-crib-Y-slot-v4'
  self.declared_initial_slots=copy.deepcopy(ROLE_NOMINAL)
