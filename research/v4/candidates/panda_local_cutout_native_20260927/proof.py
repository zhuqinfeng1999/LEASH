"""Exact frozen-model/semantic guard; no alpha-only equivalence shortcut."""
import os
os.environ['LIBERO_CONFIG_PATH']='__LEASH_ROOT__/configs/runtime/libero'
import hashlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,mujoco
from robosuite.utils.binding_utils import MjSim
from research.v4.accelerated.tabletop import create_scene
from research.v4.candidates.panda_glyph_native_pair_20260927.native_pair import arrays,exact,state
from research.v4.candidates.panda_glyph_hide_visual_20260927.helper import mesh_signature
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PARENT=ROOT/'research/v4/candidates/panda_local_cutout_fill_20260927'
EXPECTED=json.loads((PARENT/'cpu_contract.json').read_text())
BASE_XML=(PARENT/'before_scene.xml').read_text();PATCH_XML=(PARENT/'candidate_scene.xml').read_text()
ORIGINAL_FACTORY=MjSim.from_xml_string

def semantic(before,after):
 ba,aa=arrays(before),arrays(after)
 changed={k for k in ba if not exact(ba[k],aa[k])}
 if changed!=set(EXPECTED['changed_arrays']):raise RuntimeError('Unexpected compiled changed-field set '+str(sorted(changed^set(EXPECTED['changed_arrays']))))
 if EXPECTED['unexpected_nonvisual_changes']:raise RuntimeError('Parent physical proof was not accepted')
 altered={int(i)for k in('geom_pos','geom_quat','geom_size','geom_rbound','geom_aabb','geom_group')if k in changed for i in np.unique(np.argwhere(ba[k]!=aa[k])[:,0])}
 if altered!={44,45}:raise RuntimeError('Unexpected modified geom identities')
 if int(after.geom_group[45])!=5 or int(before.geom_group[45])!=1:raise RuntimeError('Wrong glyph visibility group')
 if any(int(after.geom_contype[i])or int(after.geom_conaffinity[i])for i in(44,45)):raise RuntimeError('Patched visual geoms have physical collision enabled')
 collision=np.where((before.geom_contype!=0)|(before.geom_conaffinity!=0))[0]
 fields=('geom_type','geom_dataid','geom_bodyid','geom_contype','geom_conaffinity','geom_pos','geom_quat','geom_size','geom_rbound','geom_aabb','geom_friction','geom_solref','geom_solimp')
 for k in fields:
  if not exact(getattr(before,k)[collision],getattr(after,k)[collision]):raise RuntimeError('Physical collision mismatch '+k)
 others=[]
 for i in range(before.nmesh):
  if before.mesh(i).name=='robot0_link5_vis_0':continue
  if before.mesh(i).name!=after.mesh(i).name or mesh_signature(before,i)!=mesh_signature(after,i):raise RuntimeError('Unchanged mesh differs '+before.mesh(i).name)
  others.append(before.mesh(i).name)
 for k,v in arrays(before.opt).items():
  if not exact(v,arrays(after.opt)[k]):raise RuntimeError('Physics option array differs '+k)
 for k in dir(before.opt):
  v=getattr(before.opt,k)
  if isinstance(v,(int,float,bool))and v!=getattr(after.opt,k):raise RuntimeError('Physics option scalar differs '+k)
 stat={}
 for k in('center','extent','meaninertia','meanmass','meansize'):
  a,b=np.asarray(getattr(before.stat,k)),np.asarray(getattr(after.stat,k));stat[k]={'equal':exact(a,b),'before':a.tolist(),'after':b.tolist()}
  if k!='meansize'and not exact(a,b):raise RuntimeError('Model physical/camera scale statistic differs '+k)
 # meansize is a visual-geometric summary only and never substitutes for a
 # physical equality assertion. Its measured value is explicitly retained.
 return {'passed':True,'model_arrays':len(ba),'exact_nonvisual_arrays':len(ba)-len(changed),'intentional_visual_changed_fields':sorted(changed),'expected_changed_geom_ids':sorted(altered),'collision_geoms':len(collision),'collision_fields_exact':list(fields),'other_mesh_count':len(others),'other_mesh_vertices_faces_exact':others,'physics_options_exact':True,'statistics':stat,'not_alpha_only':True}

def initialized(which,camera_observations,egl):
 if which not in('before','after'):raise ValueError(which)
 expected=mujoco.MjModel.from_xml_string(BASE_XML);chosen=mujoco.MjModel.from_xml_string(PATCH_XML if which=='after'else BASE_XML)
 if which=='after':semantic(expected,chosen)
 calls=[]
 def factory(cls,incoming):
  # Normal common constructor must still generate the pinned BASE model.
  actual=mujoco.MjModel.from_xml_string(incoming)
  bad=[k for k,v in arrays(expected).items()if not exact(v,arrays(actual)[k])]
  if bad:raise RuntimeError('Current constructor no longer matches frozen before XML '+str(bad))
  calls.append({'incoming_model_exact':True,'chosen':which})
  return ORIGINAL_FACTORY(PATCH_XML if which=='after'else BASE_XML)
 prior=MjSim.__dict__['from_xml_string']
 with patch.object(MjSim,'from_xml_string',classmethod(factory)):
  scene=create_scene('R2-01',0,camera_observations=camera_observations,render_gpu_device_id=egl)
 if MjSim.__dict__['from_xml_string']is not prior or len(calls)!=1:
  scene.close();raise RuntimeError('Production compile binding not restored/executed once')
 # Existing _reset_internal performs its normal initialization. No extra step,
 # forward or state transfer/restoration follows it.
 scene._brand_compile_calls=calls
 return scene

def active_contacts(scene):
 d=scene.sim.data._data;n=int(d.ncon)
 return {'ncon':n,'contacts':[{'geom':[int(c.geom1),int(c.geom2)],'dist':float(c.dist),'pos':np.asarray(c.pos).tolist(),'frame':np.asarray(c.frame).tolist()}for c in d.contact[:n]]}

def hashes():
 pins=json.loads((PARENT/'source_freeze.json').read_text())
 for name,h in pins.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Parent97 source drift '+name)
 for p in [PARENT/'source_freeze.json',*HERE.glob('*.py'),HERE/'cpu_contract.json',ROOT/'research/v4/candidates/panda_glyph_close_native_20260927/camera_fit.json',ROOT/'research/v4/candidates/panda_glyph_close_native_20260927/derive.py',ROOT/'research/v4/candidates/panda_glyph_native_pair_20260927/native_pair.py',ROOT/'research/v4/bridge/run_b0.py',ROOT/'envs/libero/lib/python3.11/site-packages/robosuite/utils/binding_utils.py']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));freeze=HERE/'source_freeze.json'
 if freeze.exists()and json.loads(freeze.read_text())!=pins:raise RuntimeError('Native candidate frozen drift')
 return pins
