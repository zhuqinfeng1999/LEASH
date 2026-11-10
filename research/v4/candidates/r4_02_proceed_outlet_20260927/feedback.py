"""Both methods' current public latch feedback, no tool/control changes."""
from copy import deepcopy
import time
from research.v3.bridge.b0 import _tool_result
class SharedGripFeedback:
 def __init__(self,base,scene):self.base=base;self.scene=scene;self.sensor_feedback=[]
 def __getattr__(self,name):return getattr(self.base,name)
 def get_tools_spec(self):
  specs=deepcopy(self.base.get_tools_spec())
  for s in specs:
   if s['name']=='view_env_state':s['description']=s.get('description','')+' Includes a separately timestamped current simulated grip-latch sensor; it does not describe a requested historical image.'
  return specs
 def execute_tool(self,name,args):
  result=self.base.execute_tool(name,args)
  try:latch=self.scene.public_grasp_latched()
  except (AttributeError,ValueError):latch=None
  if type(latch)is not bool:latch=None
  sensor=dict(grasp_latched=latch,hold_estimate='unknown'if latch is None else'held'if latch else'empty',source='common_simulated_grip_sensor',observed_monotonic_ns=time.monotonic_ns())
  self.sensor_feedback.append({'tool':name,**sensor})
  value=result.result
  if isinstance(value,dict):return _tool_result(name,{**value,'current_public_grip_sensor':sensor})
  return result
 def cancel_active_and_wait(self):return self.base.cancel_active_and_wait()
 def close(self):return self.base.close()
