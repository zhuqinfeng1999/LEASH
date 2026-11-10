"""Visual-only skin partition along original disconnected garment components."""
import copy,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
NAME='seated_person_male_casualsuit06'
SOURCE=ROOT/'research/upstream/makehuman-character/clothes/male_casualsuit06'
SOURCE_HASHES={'male_casualsuit06.obj':'9a86386435d7557a520320b8db4cd0ffe4755040044d1925c2b5730b864a603e',
 'male_casualsuit06.mhclo':'9cebda22a0fa57fc556a1a77723f59b176724dc7e987c01214283677444ddcc5',
 'male_casualsuit06.mhmat':'2210d331d0e319147280363e281a4e98d24b71e678dac243d1250f5d27e9169f',
 'male_casualsuit06_diffuse.png':'e9a581e0d4eb64c7a031886f2faf109a4703a3046a5fcc97ea9ce9112e99a344'}
def check_source():
 for k,h in SOURCE_HASHES.items():
  if hashlib.sha256((SOURCE/k).read_bytes()).hexdigest()!=h:raise RuntimeError('Selected garment source drift '+k)
def groups(skin):
 uv=np.asarray(list(map(float,skin.get('texcoord').split()))).reshape(-1,2)
 faces=np.asarray(list(map(int,skin.get('face').split()))).reshape(-1,3)
 # Existing actual UV island gap: flipped-shirtV<=.4048, jeansV>=.4417.
 # Require whole faces in either disconnected component, never slice a face.
 shirt=np.all(uv[faces,1]<.425,axis=1);jeans=np.all(uv[faces,1]>.425,axis=1)
 if int(shirt.sum())!=5456 or int(jeans.sum())!=1712 or not np.all(shirt^jeans):raise RuntimeError('Exact selected garment UV components changed')
 results={}
 for key,mask in [('shirt',shirt),('jeans',jeans)]:
  face_ids=np.flatnonzero(mask);vertex_ids=np.unique(faces[mask]);results[key]={'face_ids':face_ids,'vertex_ids':vertex_ids}
 if np.intersect1d(results['shirt']['vertex_ids'],results['jeans']['vertex_ids']).size:raise RuntimeError('Material partition shares a vertex/normal across components')
 if not np.array_equal(np.sort(np.concatenate([x['vertex_ids']for x in results.values()])),np.arange(len(uv))):raise RuntimeError('Original garment vertex omitted')
 return results
def patched_xml(original):
 check_source();tree=ET.fromstring(original);asset=tree.find('./asset');skin=asset.find(f"skin[@name='{NAME}']")
 if skin is None:raise RuntimeError('Selected garment skin missing')
 maps=groups(skin);vertices=np.asarray(skin.get('vertex').split()).reshape(-1,3);uv=np.asarray(skin.get('texcoord').split()).reshape(-1,2)
 faces=np.asarray(skin.get('face').split(),dtype=int).reshape(-1,3)
 original_material=skin.get('material');original_mat=asset.find(f"material[@name='{original_material}']")
 if original_mat is None or original_mat.get('texture')!=NAME+'_texture':raise RuntimeError('Selected mapped denim material changed')
 plain=NAME+'_plain_cotton_material'
 ET.SubElement(asset,'material',name=plain,rgba='.70 .71 .72 1',specular='.035',shininess='.10',reflectance='0')
 for key,mapping in maps.items():
  ids=mapping['vertex_ids'];inv=np.full(len(vertices),-1,int);inv[ids]=np.arange(len(ids))
  new=copy.deepcopy(skin);new.set('name',NAME+'_'+key)
  new.set('vertex',' '.join(vertices[ids].ravel()));new.set('texcoord',' '.join(uv[ids].ravel()))
  new.set('face',' '.join(map(str,inv[faces[mapping['face_ids']]].ravel())))
  if key=='shirt':new.set('material',plain)
  for bone in list(new.findall('bone')):
   bi=np.asarray(list(map(int,bone.get('vertid').split())));bw=np.asarray(bone.get('vertweight').split());sel=inv[bi]>=0
   if not sel.any():new.remove(bone);continue
   bone.set('vertid',' '.join(map(str,inv[bi[sel]])));bone.set('vertweight',' '.join(bw[sel]))
  asset.append(new)
 asset.remove(skin)
 return ET.tostring(tree,encoding='unicode'),{k:{a:b.tolist()for a,b in v.items()}for k,v in maps.items()}
