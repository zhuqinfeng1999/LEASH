"""One unchanged own 20-neutral support witness at the one front slot."""
from pathlib import Path
from types import FunctionType
import json
import mujoco
import numpy as np
from research.v4.candidates.r302_rear_presentation_v1_20260928 import support as old
from research.v4.candidates.r302_horizontal_short_v_20260928.copy import exact_contacts
from .scene import FrontSlotShortVCanScene
from .admit import verify

H=Path(__file__).resolve().parent
def observed_sample(scene):
    row=old.sample(scene)
    m,d=scene.backend.sim.model._model,scene.backend.sim.data._data
    e=m.site('gripper0_right_grip_site').id
    arm=[m.joint('robot0_joint'+str(i)).id for i in range(1,8)]
    controller=scene.backend.robots[0].part_controllers['right']
    contacts=exact_contacts(m,d)
    row.update(actual_arm_qpos_rad=d.qpos[m.jnt_qposadr[arm]].tolist(),
               actual_eef_m=d.site_xpos[e].tolist(),actual_eef_R=d.site_xmat[e].reshape(3,3).tolist(),
               actual_V_body_m=d.xpos[m.body('r302_v_wood_cradle').id].tolist(),
               OSC_initial_joint=np.asarray(controller.initial_joint).tolist(),
               whole_robot_can_environment_contacts=contacts)
    forces=[]
    for i,c in enumerate(d.contact[:d.ncon]):
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        forces.append({'geoms':[m.geom(int(g)).name for g in (c.geom1,c.geom2)],
                       'dist_m':float(c.dist),'contact_frame_world':c.frame.tolist(),
                       'force_contact_frame':force.tolist()})
    row['actual_all_contact_forces']=forces
    with (H/'neutral_geometry_journal.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    bad=[r for r in contacts if r['dist_m']<-.001 and not r['exact_two_soft_pad_can_material']]
    if bad:raise RuntimeError('First actual neutral hard contact >1mm: '+json.dumps(bad[0]))
    return row

def main():
    verify()
    ns=dict(old.main.__globals__,HERE=H,ROOT=H.parents[3],RearPresentedCanScene=FrontSlotShortVCanScene,
            sample=observed_sample)
    FunctionType(old.main.__code__,ns,name='main')()
if __name__=='__main__':main()
