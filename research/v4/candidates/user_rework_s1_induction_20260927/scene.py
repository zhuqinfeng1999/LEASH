"""Five own S1 factories; one shared visual/person precompile context only."""
import importlib
from contextlib import contextmanager
from unittest.mock import patch
from research.safevla import person as people
from research.v4.candidates.user_rework_p2_20260927.scene_v3 import NaturalReceivingPerson,LEFT_HAND,RIGHT_HAND
from research.v4.candidates.s1_v42_tabletop_20260925 import scene as base,scene_v3_clear_handle as clear
from .fixture import add_fixtures

CONFIG={
 'S1-01':('s1_pan_authored_complete_visual_20260927.scene','CompleteVisualScene','s1_v42_tabletop_20260925.scripts_v5_stable_pan300'),
 'S1-02':('s1_02_saucepan_t8_20260926.scene','S1SaucepanFacade','s1_v42_tabletop_20260925.scripts_v5_stable_pan300'),
 'S1-03':('s1_03_kettle_t8_20260926.scene','S1KettleFacade','s1_v42_tabletop_20260925.scripts_v3'),
 'S1-04':('s1_04_ladle_t8_20260926.scene','S1LadleFacade','s1_v42_tabletop_20260925.scripts_v3'),
 'S1-05':('s1_05_griddle_t8_20260926.scene','S1GriddleFacade','s1_05_loaded_control_20260926.scripts'),
}
PREFIX='research.v4.candidates.'
PAN_NAME=base.PAN_NAME

@contextmanager
def constructor_context():
 with patch.object(people,'Person',NaturalReceivingPerson),patch.object(base,'LEFT_HAND',LEFT_HAND),patch.object(base,'RIGHT_HAND',RIGHT_HAND),patch.object(clear,'RIGHT_HAND_CLEAR',RIGHT_HAND),patch.object(clear,'add_fixtures',add_fixtures):yield

def factory(instance):
 module,name,_=CONFIG[instance];own=getattr(importlib.import_module(PREFIX+module),name)
 class ReworkedOwnFacade(own):
  task_id=instance
  def __init__(self,*a,**kw):
   with constructor_context():super().__init__(*a,**kw)
 ReworkedOwnFacade.__name__=instance.replace('-','')+'InductionReceivingFacade'
 return ReworkedOwnFacade

def create(seed,*,instance='S1-01',camera_observations=False,render_gpu_device_id=-1):
 f=factory(instance)(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
 return f.backend,f.grasp_assist

def instructions(instance):
 if instance=='S1-01':return base.INSTRUCTIONS
 return importlib.import_module(PREFIX+CONFIG[instance][0]).INSTRUCTIONS
