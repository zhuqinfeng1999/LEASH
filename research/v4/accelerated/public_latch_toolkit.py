"""Expose one Boolean assisted-grasp sensor equally to B0 and Ours.

The sensor is a declared simulated proprioceptive latch. It provides no object
name, grasp-zone geometry, evaluator label, or target information. The same
wrapper is installed around the RPent toolkit for every paired method run.
"""
from __future__ import annotations

from research.v3.bridge.b0 import _tool_result

ENV_TOOLS = frozenset({
    'view_env_state','view_camera_meta','back_project','segment',
    'move_to','move_pose','rotate_wrist','rotate_pitch','pi0_pick',
    'pi0_doubled','release','set_gripper','hold',
})


class PublicLatchToolkit:
    def __init__(self, base, scene):
        self.base=base
        self.scene=scene

    def __getattr__(self,name):
        return getattr(self.base,name)

    def get_tools_spec(self):
        return self.base.get_tools_spec()

    def execute_tool(self,name,input_dict):
        result=self.base.execute_tool(name,input_dict)
        if name not in ENV_TOOLS:
            return result
        public=getattr(result,'result',None)
        if not isinstance(public,dict):
            return result
        # Source is the same public facade sensor supplied to the reviewer.
        latch=self.scene.public_grasp_latched()
        visible=dict(public)
        visible['grasp_latched']=latch
        return _tool_result(name,visible)

    def cancel_active_and_wait(self):
        return self.base.cancel_active_and_wait()

    def close(self):
        return self.base.close()
