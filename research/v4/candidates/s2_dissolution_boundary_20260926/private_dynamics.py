"""Narrow physical-entity consumption override; no action authorization."""
import numpy as np
from research.v4.candidates.s2_scene_20260924.private_dynamics import PrivateDynamics as OriginalDynamics,PILLS,MUG,crossed_milk

class DissolutionSafeDynamics(OriginalDynamics):
    def __init__(self):
        super().__init__();self.dissolution_detach_events=[]
    def on_step(self,scene):
        sim=scene.env.sim;objects=scene.env.env.objects_dict
        mug_xyz=np.asarray(sim.data.get_joint_qpos(objects[MUG].joints[0])[:3],float)
        detached=[]
        assist=getattr(scene,'grasp_assist',None)
        # The original entry predicate and one-dose-per-pill guard are reused.
        # Only a pill about to be consumed by that event can lose its weld.
        for name in PILLS:
            if name in self.dissolved_pills:continue
            pill=objects[name];xyz=np.asarray(sim.data.get_joint_qpos(pill.joints[0])[:3],float)
            previous=self.previous_positions.get(name,xyz)
            if not crossed_milk(mug_xyz,previous,xyz) or assist is None:continue
            matches=[(index,body,eq) for index,(zone,body,site,eq) in enumerate(assist._entries) if zone.object_body==pill.root_body]
            if len(matches)!=1:raise RuntimeError('consumed pill lacks unique declared assisted weld')
            index,body,eq=matches[0]
            active=bool(sim.data.eq_active[eq]);held=assist._held_index==index
            if active!=held:raise RuntimeError('consumed pill weld/latch bookkeeping mismatch')
            if not held:continue
            # Do not call before_control(open), alter gripper commands, release
            # other objects, or authorize an ordinary tool release. This is the
            # end of one physically consumed entity, before its hidden removal.
            sim.data.eq_active[eq]=0;assist._held_index=None
            detached.append({'sim_time':float(sim.data.time),'pill':name,'event':'consumed_pill_weld_cleared','reason':'verified_first_milk_entry_entity_consumption','weld':assist._entries[index][0].weld_name,'gripper_command_unchanged':True,'entry_xyz_m':xyz.tolist()})
        events=super().on_step(scene)
        dissolved_now={event['pill'] for event in events}
        if any(event['pill'] not in dissolved_now for event in detached):raise RuntimeError('weld cleared without matching canonical first dissolution')
        self.dissolution_detach_events.extend(detached)
        return events
