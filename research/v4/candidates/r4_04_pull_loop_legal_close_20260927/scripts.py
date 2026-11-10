"""Common real OSC/assist T1 witness, only source loop geometry/approach."""
import inspect,numpy as np
from . import check
from research.v4.candidates.r4_04_real_corner_zone_v2_20260926 import scripts as parent
from research.v4.candidates.r4_04_visible_pull_loop_20260927.scene import Scene
from research.v4.candidates.r4_04_visible_wood_riser_20260927.check import parent as geomparent
SRC=inspect.getsource(parent.run_case)
a="R=ref.site_xmat[sid].reshape(3,3).copy()"
b="pillow_R=d.xmat[m.body(item.root_body).id].reshape(3,3);X=pillow_R[:,2];Z=np.array([0.,0.,-1.]);R=np.stack([X,np.cross(Z,X),Z],axis=1)"
assert SRC.count(a)==1;SRC=SRC.replace(a,b)
a="zone=d.site_xpos[m.site('v42_adult_pillow_1_grasp_zone').id].copy();move([zone[0],zone[1],1.20],'above_actual_pillow_edge');move(zone,'close_at_actual_cloth_edge',n=180);hold(1,24);r['attached']=bool(s.grasp_assist.held)"
b="""zone_id=m.site('v42_adult_pillow_1_grasp_zone').id;zone=d.site_xpos[zone_id].copy()
   cloud=np.vstack([geom_points(m,d,g)for g in range(m.ngeom)if(m.geom(g).name or'').startswith('adult_pillow_1_')and(m.geom_contype[g]or m.geom_conaffinity[g])])
   local_gripper=np.vstack([(geom_points(m,d,g)-d.site_xpos[sid])@d.site_xmat[sid].reshape(3,3)for g in range(m.ngeom)if(m.geom(g).name or'').startswith('gripper0_')and m.geom_group[g]==0]);axis=R[:,2];extent=float(np.max((local_gripper@R.T)@axis));dist=max(.15,float(zone@axis-np.min(cloud@axis)+extent+.005));pre=zone-axis*dist
   r['actual_exposed_loop_prepoint_m']=pre.tolist();r['original_topdown_R']=R.tolist();move(pre,'above_actual_exposed_loop');move(zone,'close_actual_loop',n=180);hold(1,24);r['attached']=bool(s.grasp_assist.held)"""
assert SRC.count(a)==1;SRC=SRC.replace(a,b)
a="move([zone[0],zone[1],1.24],'actual_pillow_lift',grip=1,n=200)";b="liftgoal=np.asarray(s.raw['robot0_eef_pos']).copy();liftgoal[2]+=.12;move(liftgoal,'actual_pillow_lift',grip=1,n=200)";assert SRC.count(a)==1;SRC=SRC.replace(a,b)
a="for _ in range(n):\n   delta=goal-np.asarray(s.raw['robot0_eef_pos'])";b="for _ in range(n):\n   if name=='close_actual_loop':goal=d.site_xpos[m.site('v42_adult_pillow_1_grasp_zone').id].copy()\n   delta=goal-np.asarray(s.raw['robot0_eef_pos'])";assert SRC.count(a)==1;SRC=SRC.replace(a,b)
ns=dict(vars(parent));ns.update(T1R4AdultPillowScene=Scene,geom_points=geomparent.geom_points);exec(compile(SRC,__file__,'exec'),ns);run_case=ns['run_case']
