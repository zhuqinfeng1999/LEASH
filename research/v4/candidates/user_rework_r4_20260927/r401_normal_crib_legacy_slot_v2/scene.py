"""Own legacy carrier; exact-count full-crib XML/initial-slot injection only."""
import ast,copy,inspect,textwrap
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from research.v4.accelerated import r4_native as original
from research.v4.candidates.r4_01_raised3mm_first_risk_20260926 import scene as own
from ..r402_drop_side_v5 import scene as furniture
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
ROLE_NOMINAL={'item':[-.06,-.16],'crib':[-.48,.90]}
ORIGINAL_INIT=original.Scene.__init__;ORIGINAL_LOAD=original.load_instance

def configuration(instance_id):
 if instance_id!='R4-01':raise ValueError('Only own R401 registered bag')
 instance=ORIGINAL_LOAD(instance_id);crib=copy.deepcopy(instance.crib)
 crib.update(xml=str(furniture.XML),initial_z_m=.64,footprint_half_xy_m=[.632,.332],inside_half_xy_m=[.60,.30],inside_bottom_z_m=.040,inside_top_z_m=.45)
 return replace(instance,crib=crib)

def render_bddl(instance):
 source=original.render_bddl(instance)
 old='(crib_start (:target main_table) (:ranges ((-.065 .155 -.055 .165))))'
 new='(crib_start (:target main_table) (:ranges ((-.485 .895 -.475 .905))))'
 if source.count(old)!=1:raise RuntimeError('Own initial crib BDDL range changed')
 return source.replace(old,new)

INIT_SOURCE=textwrap.dedent(inspect.getsource(ORIGINAL_INIT))
OLD='qpos[:3] = [-.06, nominal_y, asset["initial_z_m"]]'
NEW='qpos[:3] = [*ROLE_NOMINAL[role], asset["initial_z_m"]]'
if INIT_SOURCE.count(OLD)!=1:raise RuntimeError('Own final physical slot assignment changed')
INIT_SOURCE=INIT_SOURCE.replace(OLD,NEW)
INIT_NS={**original.__dict__,'ROLE_NOMINAL':ROLE_NOMINAL,'load_instance':configuration,'render_bddl':render_bddl}
exec(compile(ast.parse(INIT_SOURCE),__file__+':unchanged-staged-carrier-one-slot-line','exec'),INIT_NS)
SELECTED_INIT=INIT_NS['__init__']

class Scene(own.Scene):
 def __init__(self,*args,**kwargs):
  # Original own constructor still selects CPU/renderer and all compile/reset
  # stages. No post-construction relocation, added steps or source weld.
  with patch.object(original.Scene,'__init__',SELECTED_INIT),patch.object(own,'XML',furniture.XML):
   super().__init__(*args,**kwargs)
  self.layout_version='R401-original-bag-staged-carrier-normal-floor-crib-slot-v2'
  self.declared_initial_slots=copy.deepcopy(ROLE_NOMINAL)
