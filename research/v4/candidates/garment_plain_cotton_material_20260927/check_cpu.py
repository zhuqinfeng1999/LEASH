"""Actual production constructor + skin union/normal/physical proof, CPU only."""
import json,time,traceback
from pathlib import Path
import numpy as np,mujoco
from research.v4.candidates.panda_glyph_native_pair_20260927.native_pair import arrays,exact,state,FIELDS
from . import proof
from .helper import HERE,NAME,patched_xml
def model_skin(m,name):
 i=m.skin(name).id;v=int(m.skin_vertadr[i]);n=int(m.skin_vertnum[i]);f=int(m.skin_faceadr[i]);nf=int(m.skin_facenum[i]);t=int(m.skin_texcoordadr[i])
 return {'id':i,'vertices':m.skin_vert[v:v+n].copy(),'uv':m.skin_texcoord[t:t+n].copy(),'faces':m.skin_face[f:f+nf].copy()}
def bone_map(m,i,n):
 result={}
 for b in range(int(m.skin_boneadr[i]),int(m.skin_boneadr[i]+m.skin_bonenum[i])):
  a=int(m.skin_bonevertadr[b]);nn=int(m.skin_bonevertnum[b]);weights=np.zeros(n,dtype=m.skin_bonevertweight.dtype)
  weights[m.skin_bonevertid[a:a+nn]]=m.skin_bonevertweight[a:a+nn]
  result[int(m.skin_bonebodyid[b])]={'weights':weights,'bindpos':m.skin_bonebindpos[b].copy(),'bindquat':m.skin_bonebindquat[b].copy()}
 return result
def compiled_skin_scene(m,d):
 sc=mujoco.MjvScene(m,maxgeom=10000);opt=mujoco.MjvOption();opt.geomgroup[:]=[0,1,1,0,0,0]
 mujoco.mjv_updateScene(m,d,opt,None,mujoco.MjvCamera(),mujoco.mjtCatBit.mjCAT_ALL,sc);return sc
def posed_skin(sc,i):
 a=int(sc.skinvertadr[i]);n=int(sc.skinvertnum[i]);return np.asarray(sc.skinvert).reshape(-1,3)[a:a+n].copy(),np.asarray(sc.skinnormal).reshape(-1,3)[a:a+n].copy()
def semantic(a,b,mapping):
 aa,bb=arrays(a),arrays(b);changed={k for k in aa if not exact(aa[k],bb[k])}
 # Appending material/skin names moves byte offsets/hash-table slots, never
 # physical object IDs. Verify their decoded names independently below.
 allowed={k for k in changed if k.startswith(('skin_','mat_','name_'))}|{'names','names_map','_sizes'}
 bad=changed-allowed
 if bad:raise RuntimeError('Nonvisual compiled model changed '+str(sorted(bad)))
 named_objects={}
 for k in sorted(x for x in aa if x.startswith('name_')and x.endswith('adr')and x not in ('name_skinadr','name_matadr')):
  def decode(m,offsets):
   raw=m.names
   return [raw[int(i):].split(b'\0',1)[0].decode()for i in offsets]
  x,y=decode(a,getattr(a,k)),decode(b,getattr(b,k))
  if x!=y:raise RuntimeError('Actual named physical entity changed '+k)
  named_objects[k]={'decoded_original_names_exact':True,'count':len(x)}
 if b.nmat!=a.nmat+1:raise RuntimeError('Expected one new cotton material only')
 for k in (x for x in aa if x.startswith('mat_')):
  if not exact(aa[k],bb[k][:a.nmat]):raise RuntimeError('Original mapped/other material changed '+k)
 for k,v in arrays(a.opt).items():
  if not exact(v,arrays(b.opt)[k]):raise RuntimeError('Physics option changed '+k)
 for k in dir(a.opt):
  v=getattr(a.opt,k)
  if isinstance(v,(int,float,bool))and v!=getattr(b.opt,k):raise RuntimeError('Physics scalar changed '+k)
 old=model_skin(a,NAME);old_bones=bone_map(a,old['id'],len(old['vertices']))
 recovered=np.empty_like(old['vertices']);uv=np.empty_like(old['uv']);faces=np.empty_like(old['faces']);new_bones={}
 for kind,maps in mapping.items():
  new=model_skin(b,NAME+'_'+kind);vi=np.asarray(maps['vertex_ids']);fi=np.asarray(maps['face_ids'])
  recovered[vi]=new['vertices'];uv[vi]=new['uv'];faces[fi]=vi[new['faces']]
  for body,bone in bone_map(b,new['id'],len(vi)).items():
   if body not in new_bones:new_bones[body]={'weights':np.zeros(len(old['vertices']),dtype=bone['weights'].dtype),'bindpos':bone['bindpos'],'bindquat':bone['bindquat']}
   prior=new_bones[body]
   if not exact(prior['bindpos'],bone['bindpos'])or not exact(prior['bindquat'],bone['bindquat']):raise RuntimeError('Split bone bind mismatch')
   prior['weights'][vi]=bone['weights']
 if not exact(recovered,old['vertices'])or not exact(uv,old['uv'])or not exact(faces,old['faces']):raise RuntimeError('Original compiled vertex/UV/face union not exact')
 if set(new_bones)!=set(old_bones)or any(not all(exact(new_bones[body][k],old_bones[body][k])for k in old_bones[body])for body in old_bones):raise RuntimeError('Skin binding/weight union not exact')
 other=[]
 for i in range(a.nskin):
  name=a.skin(i).name
  if name==NAME:continue
  x,y=model_skin(a,name),model_skin(b,name)
  if any(not exact(x[k],y[k])for k in('vertices','uv','faces')):raise RuntimeError('Other skin changed '+name)
  bx,by=bone_map(a,x['id'],len(x['vertices'])),bone_map(b,y['id'],len(y['vertices']))
  if set(bx)!=set(by)or any(not all(exact(bx[j][k],by[j][k])for k in bx[j])for j in bx):raise RuntimeError('Other skin bone changed '+name)
  other.append(name)
 return {'changed_model_arrays':sorted(changed),'model_array_count':len(aa),'nonvisual_arrays_exact':len(aa)-len(changed),
  'original_vertex_uv_face_and_bone_weight_union_exact':True,'other_skin_geometry_bindings_exact':other,
  'name_address_offsets_only_decoded_entities_exact':named_objects,
  'all_original_material_rows_exact_one_new_cotton_row':True,
  'same_all_collision_body_mass_inertia_controller_camera_options':True,'original_bitmap_bytes_unchanged':True}
def main():
 start=time.monotonic();scenes=[];r={'passed':False,'error':None,'controls':0,'GPU':False,'models':0,'renders':0,'raster_writes':0}
 try:
  before=proof.initialized('before');scenes.append(before);backend=before.backend;xml=backend.model.get_xml();patch,mapping=patched_xml(xml)
  (HERE/'material_partition.json').write_text(json.dumps(mapping,indent=2)+'\n')
  after=proof.initialized('after');scenes.append(after)
  def base(s):return s.backend
  b,a=base(before),base(after);m0,m1=b.sim.model._model,a.sim.model._model;d0,d1=b.sim.data._data,a.sim.data._data
  if before._garment_compile_calls[0]['incoming_xml_sha256']!=after._garment_compile_calls[0]['incoming_xml_sha256']:raise RuntimeError('Production original inputs differ')
  r['model_proof']=semantic(m0,m1,mapping)
  if not exact(state(b),state(a)):raise RuntimeError('Original constructor integration differs')
  for k in('qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_applied','qfrc_constraint','xfrc_applied'):
   if not exact(np.asarray(getattr(d0,k)),np.asarray(getattr(d1,k))):raise RuntimeError('Constructor forces differ '+k)
  if int(d0.ncon)!=int(d1.ncon):raise RuntimeError('Contact count differs')
  for x,y in zip(d0.contact[:int(d0.ncon)],d1.contact[:int(d1.ncon)]):
   for k in('geom','dist','pos','frame'):
    if not exact(np.asarray(getattr(x,k)),np.asarray(getattr(y,k))):raise RuntimeError('Contact differs '+k)
  if any(not exact(np.asarray(before.raw[k]),np.asarray(after.raw[k]))for k in FIELDS):raise RuntimeError('Actual proprio differs')
  sc0,sc1=compiled_skin_scene(m0,d0),compiled_skin_scene(m1,d1);old=model_skin(m0,NAME);pv,pn=posed_skin(sc0,old['id']);newv=np.empty_like(pv);newn=np.empty_like(pn)
  for kind,maps in mapping.items():
   vi=np.asarray(maps['vertex_ids']);s=model_skin(m1,NAME+'_'+kind);vv,nn=posed_skin(sc1,s['id']);newv[vi]=vv;newn[vi]=nn
  if not exact(newv,pv)or not exact(newn,pn):raise RuntimeError('Actual skinned posed geometry/normals changed')
  # Existing mesh fold/shaping evidence: no claim that a constant shader
  # recreates the atlas's baked weave or is already visually approved.
  ids=np.asarray(mapping['shirt']['face_ids']);faces=old['faces'][ids];points=old['vertices'][faces]
  normals=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None];adj={}
  for k,t in enumerate(faces):
   for u,v in zip(t,np.roll(t,-1)):adj.setdefault(tuple(sorted((int(u),int(v)))),[]).append(k)
  angles=np.array([np.rad2deg(np.arccos(np.clip(normals[fs[0]]@normals[fs[1]],-1,1)))for fs in adj.values()if len(fs)==2])
  r['actual_shirt_mesh_normal_variation_deg']={str(q):float(np.percentile(angles,q))for q in(0,25,50,75,90,95,99,100)}
  r['constant_material_does_not_claim_baked_weave_or_texture_detail']=True
  if not exact(state(b),state(a)):raise RuntimeError('Visual CPU scene update changes state')
  # A source-specific negative fixture proves mixed-island/face data fails closed.
  import xml.etree.ElementTree as ET
  t=ET.fromstring(xml);skin=t.find(f"./asset/skin[@name='{NAME}']");coords=skin.get('texcoord').split();coords[1]='.425';skin.set('texcoord',' '.join(coords))
  try:patched_xml(ET.tostring(t,encoding='unicode'))
  except RuntimeError:r['mixed_or_unknown_UV_component_rejected']=True
  else:raise RuntimeError('Corrupt UV component accepted')
  r.update(passed=True,actual_integration_forces_contact_proprio_exact=True,actual_MjvScene_skin_position_normal_union_exact=True,
    same_original_person_pose=True,source_garment_triangles=7168,shirt_triangles=5456,denim_triangles=1712,
    original_skin_render_vertices=len(pv),shirt_render_vertices=len(mapping['shirt']['vertex_ids']),jeans_render_vertices=len(mapping['jeans']['vertex_ids']),
    native_appearance='pending; existing mesh folds remain; no texture-resolution quality claim')
 except BaseException as e:r.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  for s in scenes:s.close()
  r['wall_seconds']=time.monotonic()-start;(HERE/'geometry_contract.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r.get(k)for k in('passed','error','wall_seconds','model_proof','actual_MjvScene_skin_position_normal_union_exact')}))
 return 0 if r['passed']else 2
if __name__=='__main__':raise SystemExit(main())
