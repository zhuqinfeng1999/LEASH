"""Future opt-in appearance patch. Never imported by current common environments."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
SOURCE='envs/libero/lib/python3.11/site-packages/robosuite/models/assets/robots/panda/obj_meshes/link5_vis/link5_vis_1.obj'
SHA='1d4089d9f2d26ef677ff42b27a2cb4ba9691a8fcd19ac665ac5203b23d22b876'

def mesh_signature(m,mid):
 a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);b=int(m.mesh_faceadr[mid]);k=int(m.mesh_facenum[mid])
 h=hashlib.sha256()
 for v in(m.mesh_vert[a:a+n],m.mesh_face[b:b+k]):h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
 return h.hexdigest()

def identify(m,prefix='robot0_'):
 if hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()!=SHA:raise RuntimeError('Original official glyph mesh source changed')
 expected=prefix+'link5_vis_1';mids=[i for i in range(m.nmesh)if m.mesh(i).name==expected]
 if len(mids)!=1:raise RuntimeError('Expected exactly one named glyph mesh')
 mid=mids[0];gids=[i for i in range(m.ngeom)if int(m.geom_type[i])==7 and int(m.geom_dataid[i])==mid]
 if len(gids)!=1:raise RuntimeError('Expected exactly one glyph visual geom')
 g=gids[0]
 if m.body(int(m.geom_bodyid[g])).name!=prefix+'link5':raise RuntimeError('Glyph attached to unexpected body')
 mat=int(m.geom_matid[g])
 if mat<0 or m.mat(mat).name!=prefix+'Shell_001_001_003':raise RuntimeError('Glyph material identity changed')
 if int(m.geom_contype[g])!=0 or int(m.geom_conaffinity[g])!=0 or int(m.geom_group[g])!=1:raise RuntimeError('Glyph is no longer the identified non-collision visual geom')
 c=json.loads((HERE/'contract.json').read_text())
 if mesh_signature(m,mid)!=c['compiled_glyph_mesh_sha256']:raise RuntimeError('Compiled glyph topology/vertices changed')
 return g

def hide_glyph(m,prefix='robot0_'):
 """Validate everything before writing one geom alpha; never alters materials."""
 g=identify(m,prefix)
 prior=m.geom_rgba[g].copy()
 if not np.isfinite(prior).all()or not 0<=prior[3]<=1:raise RuntimeError('Invalid glyph RGBA')
 m.geom_rgba[g,3]=0.
 return dict(geom_id=g,geom_name=m.geom(g).name,mesh_name=m.mesh(int(m.geom_dataid[g])).name,body_name=m.body(int(m.geom_bodyid[g])).name,alpha_before=float(prior[3]),alpha_after=float(m.geom_rgba[g,3]),changed_field='geom_rgba[glyph,3]',appearance_only=True,physics_controls=0,native_visibility='pending_root_native_view_proof')
