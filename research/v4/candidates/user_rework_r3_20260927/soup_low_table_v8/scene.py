"""Own normal R302 top-grasp can, full-size front-facing oven on table."""
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as base
from research.v4.candidates.r3_03_t1_derived_20260926 import scene as configured
from research.v4.candidates.r3_02_cc0_soup_config_20260926 import scene as own
from research.v4.candidates.user_rework_r3_20260927.scene import FrontFixtureEnv
from research.v4.candidates.user_rework_r3_20260927.top_handle_prepark_v6.scene import PARK
H=Path(__file__).resolve().parent;ROOT=H.parents[4]
class SoupLowTableScene(base.R3TabletopFacade):
 task_id='R3-02';item_name='steel_thermos_1';oven_name='microwave_1'
 instruction='Put the item on the left into the microwave and close the door to heat it up.'
 layout_version='R302-own-can-normal-table-front-open-v8'
 def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
  bindings=(tabletop.ObjectBinding(self.item_name,str(own.HERE/'assets/steel_soup_can.xml'),'L',z_offset=.0575,grasp_local=(0.,0.,.042),xy_offset=(-.14,0.)),tabletop.ObjectBinding(self.oven_name,str(ROOT/'research/v4/candidates/r3_v42_tabletop_20260925/microwave_fullsize_180.xml'),'R',fixed=True,xy_offset=(.10,-.24)))
  def instance():
   with patch.object(configured,'HERE',own.HERE):return configured.instance()
  with patch.multiple(base,TASK=self.task_id,ITEM=self.item_name,_BINDINGS=bindings,_R3FixtureEnv=FrontFixtureEnv,candidate_instance=instance),patch.object(tabletop,'PARK_QPOS',PARK):super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
