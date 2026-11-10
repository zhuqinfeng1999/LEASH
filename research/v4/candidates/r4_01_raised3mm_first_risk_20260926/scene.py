"""Original R401 carrier/controller with common faithful raised doll asset."""
import copy,os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import r4_native as original
from research.v4.candidates.r4_crib_doll_raised3mm_20260926.candidate import XML
from research.v4.candidates.r3_v42_tabletop_20260925.scene import R3TabletopFacade
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
class Scene(original.Scene):
 task_id='R4-01';item_name='item_1';target_name='crib_1';instruction='Put the item on the left into the crib.'
 def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
  os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'configs/runtime/libero'))
  from libero.libero import envs
  from libero.libero.envs.env_wrapper import ControlEnv
  real_load=original.load_instance;real_renderer=envs.OffScreenRenderEnv
  def load(instance_id):
   instance=real_load(instance_id);crib=copy.deepcopy(instance.crib);crib['xml']=str(XML);return replace(instance,crib=crib)
  def backend(**kwargs):
   kwargs.update(render_gpu_device_id=render_gpu_device_id)
   if camera_observations:return real_renderer(**kwargs)
   kwargs.update(use_camera_obs=False,has_offscreen_renderer=False)
   return ControlEnv(**kwargs)
  with patch.object(original,'load_instance',load),patch.object(envs,'OffScreenRenderEnv',backend):super().__init__('R4-01',int(seed))
  self.backend=self.env.env;self.initial_item_z=float(self.env.sim.data.get_joint_qpos(self.backend.objects_dict[self.item_name].joints[0])[2])
 capture_agent=R3TabletopFacade.capture_agent
