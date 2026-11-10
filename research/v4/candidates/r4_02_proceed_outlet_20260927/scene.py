"""Same R402 physics; explicitly shared public simulated grip latch."""
from research.v4.candidates.r4_family_method_config_20260927.config import resolve,binding
Base=resolve(binding('R4-02').factory)
class Scene(Base):
 public_sensor_revision='R402-two-RGBD640-proprio-common-simulated-latch-v1'
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw)
  self.source_scene_xml=self.backend.model.get_xml()
 def public_grasp_latched(self):
  # Declared simulated sensor, same accessor/packet for B0 and Ours.
  # No object identity, zone, weld transform or evaluator flag is exposed.
  return bool(self.grasp_assist.held)
 def capture_agent(self):
  p=super().capture_agent()
  p['robot_state']['grasp_latched']=self.public_grasp_latched()
  return p
