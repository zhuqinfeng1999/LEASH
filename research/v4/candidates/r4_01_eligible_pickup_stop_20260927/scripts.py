"""Offline expert only: stop descent at actual original pickup eligibility."""
import numpy as np
from research.v4.candidates.r4_01_raised3mm_first_risk_20260926 import scripts as parent
from research.v4.candidates.r4_03_neck_layout_readonly_20260927.check_geometry import geometry
from research.v4.candidates.r4_04_side_pinch_clearance_v2_20260927.derive import geom_points

def pickup_geometry(scene):
 sim=scene.env.sim;m,d=sim.model._model,sim.data._data;g=scene.grasp_assist;zone=m.site('v42_r4_bag_grasp').id
 geo=geometry(m,d,zone,g._center,g._left,g._right)
 shared=bool(g._eligible(zone,(.015,.015,.020)))
 if shared!=geo['eligible']:raise RuntimeError('offline witness differs from unchanged common eligibility')
 tid=m.geom('table_collision').id;top=float(d.geom_xpos[tid,2]+m.geom_size[tid,2]);rows=[]
 for name in ('gripper0_right_finger1_collision','gripper0_right_finger2_collision'):
  gi=m.geom(name).id;verts=geom_points(m,d,gi);rows.append(dict(geom=name,actual_world_vertex_bounds_m=[verts.min(0).tolist(),verts.max(0).tolist()],lowest_vertex_minus_table_top_m=float(verts[:,2].min()-top)))
 ct=[]
 for c in d.contact[:d.ncon]:
  ns=[m.geom(int(i)).name or''for i in(c.geom1,c.geom2)]
  if 'table_collision'in ns and any(n.startswith(('robot0_','gripper0_'))for n in ns):ct.append(dict(geoms=ns,distance_m=float(c.dist)))
 clean=all(x['lowest_vertex_minus_table_top_m']>0 for x in rows)and not ct
 return dict(sequence=scene.sequence,time_s=float(d.time),real_original_geometry=geo,common_private_eligible=shared,complete_hard_fingers=rows,actual_table_top_m=top,robot_table_contacts=ct,hard_finger_table_clear=clean,eligible_stop=bool(shared and clean))

SOURCE=parent.SOURCE
ANCHOR='''            for _ in range(limit):
                error = dest - np.asarray(scene.raw["robot0_eef_pos"], float)'''
REPLACEMENT='''            for _ in range(limit):
                if stage == 'descend':
                    witness=pickup_geometry(scene)
                    report.setdefault('offline_pickup_geometry_checks',[]).append(witness)
                    if witness['eligible_stop']:
                        report['actual_preclose_geometry']=witness
                        report['descent_stop_cause']='first_original_between_finger_eligibility_with_clean_full_hard_finger_table_geometry'
                        break
                error = dest - np.asarray(scene.raw["robot0_eef_pos"], float)'''
if SOURCE.count(ANCHOR)!=1:raise RuntimeError('single original expert descend loop anchor drift')
SOURCE=SOURCE.replace(ANCHOR,REPLACEMENT)
namespace={**parent.namespace,'pickup_geometry':pickup_geometry}
exec(compile(SOURCE,__file__,'exec'),namespace)
def run_case(seed,case):
 try:r=namespace['run_case'](case,seed)
 except parent.ActualRiskStop as stop:r=stop.report
 r['passed']=bool(r['passed']and not r['unexpected_risk_contacts'])
 if case=='naive':r['passed']=bool(r['passed']and r.get('actual_preclose_geometry',{}).get('eligible_stop',False)and r.get('attached_after_close',False)and r.get('held_after_lift',False)and r.get('stop_witness',{}).get('real_original_critical',False))
 return r
