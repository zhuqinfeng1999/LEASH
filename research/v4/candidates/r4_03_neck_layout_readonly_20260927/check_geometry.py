"""One copied-state geometry candidate; no actual steps or replay of old path."""
import copy,json,hashlib
from pathlib import Path
import mujoco,numpy as np
from research.v4.candidates.r4_03_neck_over_block_v2_20260926.scene import T1R4BalloonScene
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OLD=ROOT/'research/v4/candidates/r4_03_neck_over_block_v2_20260926'
DIAG=ROOT/'research/v4/candidates/r4_03_readonly_reach_diagnostic_20260926/report.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(m,d):
 x=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,x,mujoco.mjtState.mjSTATE_INTEGRATION);return x
def geometry(m,d,zone,center,left,right):
 p=d.site_xpos[zone];c=d.site_xpos[center];l=d.site_xpos[left];r=d.site_xpos[right];gap=np.linalg.norm(r-l);along=np.dot(p-l,(r-l)/gap);loc=d.site_xmat[center].reshape(3,3).T@(p-c)
 return dict(zone_world_m=p.tolist(),zone_gripper_local_m=loc.tolist(),pad_gap_m=float(gap),zone_along_gap_m=float(along),between_pads=bool(0<=along<=gap),within_original_tolerance=bool(np.all(np.abs(loc)<=np.array([.015,.015,.020]))),eligible=bool(0<=along<=gap and np.all(np.abs(loc)<=np.array([.015,.015,.020]))))
def contacts(m,d):
 rows=[]
 for c in d.contact[:d.ncon]:
  ns=[m.geom(int(i)).name or ''for i in(c.geom1,c.geom2)]
  if not any(n.startswith(('robot0_','gripper0_','deflated_balloon_1_','balloon_board_1_'))for n in ns):continue
  rows.append(dict(pair=ns,distance_m=float(c.dist),position_m=c.pos.tolist(),robot=any(n.startswith(('robot0_','gripper0_'))for n in ns),balloon=any(n.startswith('deflated_balloon_1_')for n in ns),block=any(n.startswith('balloon_board_1_')for n in ns),table='table_collision'in ns))
 return rows
def main():
 out=HERE/'report.json'
 if out.exists():raise ValueError('one immutable candidate geometry only')
 proposal=json.loads((HERE/'proposal.json').read_text());snap=json.loads(DIAG.read_text())['physical_snapshot'];old=json.loads((OLD/'qualification3/naive_seed0.json').read_text());s=T1R4BalloonScene(0,camera_observations=False);original=s.backend.sim;origm,origd=original.model._model,original.data._data;original_state=state(origm,origd);original_arrays={k:sha_array(getattr(origm,k))for k in('geom_size','geom_pos','body_mass','body_inertia','site_pos')}
 report=dict(kind='COPY_MJDATA_ONE_NECK_LAYOUT_NOT_PHYSICAL_QUALIFICATION',proposal=proposal,source_sha256={},error=None,simulation_steps=0,trajectory_replays=0,models=[],gpu=False)
 try:
  m=copy.deepcopy(origm);d=mujoco.MjData(m);mujoco.mj_setState(m,d,original_state,mujoco.mjtState.mjSTATE_INTEGRATION)
  for j in snap['joints']:jid=m.joint(j['name']).id;d.qpos[m.jnt_qposadr[jid]]=j['q_rad']
  g=s.grasp_assist;center,left,right=g._center,g._left,g._right
  # Exact prismatic analytic reconstruction from each SAVED real pad, not a
  # scan of alternative grasp positions. The old raw omitted finger qpos.
  recovered=[]
  for sid,target in [(left,np.asarray(snap['left_pad_world_m'])),(right,np.asarray(snap['right_pad_world_m']))]:
   name='gripper0_right_finger_joint1'if sid==left else'gripper0_right_finger_joint2'
   jid=m.joint(name).id;adr=m.jnt_qposadr[jid];d.qpos[adr]=0;mujoco.mj_forward(m,d);p0=d.site_xpos[sid].copy();d.qpos[adr]=.01;mujoco.mj_forward(m,d);axis=(d.site_xpos[sid]-p0)/.01;q=float(axis@(target-p0)/(axis@axis));d.qpos[adr]=q;mujoco.mj_forward(m,d)
   recovered.append(dict(joint=name,qpos_rad_or_m=q,saved_pad_residual_m=float(np.linalg.norm(d.site_xpos[sid]-target)),in_joint_range=bool(m.jnt_range[jid,0]-1e-10<=q<=m.jnt_range[jid,1]+1e-10)))
  mujoco.mj_forward(m,d);grip_resid=float(np.linalg.norm(d.site_xpos[center]-snap['grip_center_world_m']));pad_resids=[float(np.linalg.norm(d.site_xpos[sid]-np.array(snap[key])))for sid,key in[(left,'left_pad_world_m'),(right,'right_pad_world_m')]]
  if grip_resid>1e-8 or max(pad_resids)>1e-8:raise RuntimeError('saved actual arm/pad reconstruction mismatch')
  item=s.backend.objects_dict[s.item_name];board=s.backend.objects_dict['balloon_board_1'];j=m.joint(item.joints[0]).id;adr=m.jnt_qposadr[j]
  # Final center is saved. Final orientation was NOT saved: prior close-stage
  # quaternion is explicitly approximate; only robot reconstruction is exact.
  q=np.asarray(old['stages'][-1]['item_qpos']);q[:3]=old['evaluator']['last_truth']['item_center_m'];d.qpos[adr:adr+7]=q;mujoco.mj_forward(m,d)
  zone=m.site('v42_deflated_balloon_1_grasp_zone').id;old_approx=geometry(m,d,zone,center,left,right)
  # ONE before-reset normal layout proposal, never applied to original data.
  q[:3]+=[proposal['balloon_source_shift_x_m'],0,proposal['balloon_raise_from_previous_supported_pose_m']];q[3:]=[1,0,0,0];d.qpos[adr:adr+7]=q;d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j]+6]=0
  m.site_pos[zone]=proposal['balloon_grasp_local_m']
  bj=m.joint(board.joints[0]).id;ba=m.jnt_qposadr[bj];d.qpos[ba:ba+7]=[*q[:2],.9+proposal['wood_size_m'][2]/2,1,0,0,0];d.qpos[ba]+=proposal['wood_shift_local_x_relative_balloon_m'];d.qvel[m.jnt_dofadr[bj]:m.jnt_dofadr[bj]+6]=0
  for name in board.contact_geoms:
   gid=m.geom(name).id
   if name.endswith('board_collision'):m.geom_size[gid]=np.array(proposal['wood_size_m'])/2
  for gid in range(m.ngeom):
   if(m.geom(gid).name or '').startswith('balloon_board_1_')and int(m.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_BOX):m.geom_size[gid]=np.array(proposal['wood_size_m'])/2
  bid=m.body(board.root_body).id;mass=proposal['wood_mass_kg'];x,y,z=proposal['wood_size_m'];m.body_mass[bid]=mass;m.body_inertia[bid]=mass/12*np.array([y*y+z*z,x*x+z*z,x*x+y*y]);mujoco.mj_forward(m,d)
  candidate=geometry(m,d,zone,center,left,right);cts=contacts(m,d)
  robot_environment=[c for c in cts if c['robot']and(c['table']or c['block'])and c['distance_m']<-.001]
  hard_balloon_overlap=[c for c in cts if c['robot']and c['balloon']and c['distance_m']<-.001]
  report.update(exact_robot_reconstruction=dict(finger_recovery=recovered,center_residual_m=grip_resid,pad_residuals_m=pad_resids),old_recorded_geometry=snap,old_approximate_balloon_geometry=old_approx,
                old_final_balloon_quaternion_missing=True,candidate_geometry=candidate,candidate_contacts=cts,robot_environment_deep_contacts=robot_environment,robot_balloon_deep_contacts=hard_balloon_overlap,
                grasp_point_inside_authored_folded_neck=True,authored_region_basis='Original folded_neck_collision box spans x13..37mm, y±4mm, z−1..3mm; (25,0,1)mm is its genuine material center. No lip outer-edge point, clip or object enlargement.',
                copied_model_changes=['private folded-neck site','normal support block 18→45mm and density-preserving mass/inertia','balloon/source/block normal before-reset pose'],
                geometry_candidate_passed=candidate['eligible']and not robot_environment and not hard_balloon_overlap,
                physical_stability_reachability_assisted_lift_unproven=True)
 except BaseException as exc:report['error']=f'{type(exc).__name__}:{exc}';report['geometry_candidate_passed']=False
 finally:
  report['original_full_integration_unchanged']=bool(np.array_equal(original_state,state(origm,origd)));report['original_model_arrays_unchanged']=original_arrays=={k:sha_array(getattr(origm,k))for k in original_arrays};s.close()
  paths=[DIAG,OLD/'qualification3/naive_seed0.json',OLD/'assets/balloon.xml',OLD/'assets/board.xml',OLD/'scene.py',OLD/'config_v1.json',HERE/'proposal.json',Path(__file__)]
  report['source_sha256']={str(p.relative_to(ROOT)):sha(p)for p in paths};out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k]for k in('geometry_candidate_passed','error','original_full_integration_unchanged','original_model_arrays_unchanged')}))
def sha_array(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
if __name__=='__main__':main()
