"""Only offline final-approach steering tracks the same current physical rim."""
import inspect
import numpy as np
from research.v4.candidates.r4_05_inclined_identity_v2_20260926 import scripts as parent
SOURCE=inspect.getsource(parent.run_case)
ANCHOR="zone=d.site_xpos[m.site('v42_magnetic_beads_box_1_grasp_zone').id].copy();move([zone[0],zone[1],1.13],'above_box_end_wall');move(zone,'close_at_real_box_end_wall',n=150,clip=.005);hold(1,24);attached=bool(s.grasp_assist.held)"
REPLACEMENT="""zone_id=m.site('v42_magnetic_beads_box_1_grasp_zone').id
   zone=d.site_xpos[zone_id].copy();move([zone[0],zone[1],1.13],'above_box_end_wall')
   first=e.samples
   for _ in range(150):
    # Offline known-script guidance only: no private site goes to a model/tool.
    zone=d.site_xpos[zone_id].copy();delta=zone-np.asarray(s.raw['robot0_eef_pos'])
    if np.linalg.norm(delta)<.007:break
    a=np.zeros(7);a[:3]=np.clip(delta,-.005,.005)/.05;a[6]=-1;command(a)
   zone=d.site_xpos[zone_id].copy()
   r['stages'].append({'name':'close_at_current_real_box_end_wall','steps':e.samples-first,'error_m':float(np.linalg.norm(zone-s.raw['robot0_eef_pos'])),'eef_m':np.asarray(s.raw['robot0_eef_pos']).tolist(),'item_qpos':sim.data.get_joint_qpos(item.joints[0]).copy().tolist(),'held':bool(s.grasp_assist.held),'offline_current_same_site_target_m':zone.tolist()})
   hold(1,24);attached=bool(s.grasp_assist.held)"""
if SOURCE.count(ANCHOR)!=1:raise ValueError('parent grasp-path anchor changed')
SOURCE=SOURCE.replace(ANCHOR,REPLACEMENT)
namespace=dict(vars(parent));exec(compile(SOURCE,__file__,'exec'),namespace)
run_case=namespace['run_case']
