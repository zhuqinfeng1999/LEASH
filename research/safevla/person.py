"""CC0 MakeHuman mesh/weights, posed as one complete seated person.

MuJoCo skin is visual. Kinematic bone capsules independently participate in
contacts. Scene owns all mocap updates; no avatar state is sent to perception.
"""
from pathlib import Path
import json
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'research/upstream/makehuman-assets'

def fmt(a):return ' '.join(f'{x:.7g}' for x in np.asarray(a).ravel())
def quat(r):
 q=Rotation.from_matrix(r).as_quat();return q[[3,0,1,2]]
def between(a,b):
 a=np.asarray(a)/np.linalg.norm(a);b=np.asarray(b)/np.linalg.norm(b)
 cross=np.cross(a,b);n=np.linalg.norm(cross)
 if n<1e-8:return np.eye(3) if np.dot(a,b)>0 else Rotation.from_rotvec(np.pi*np.array([0,1,0])).as_matrix()
 return Rotation.from_rotvec(cross/n*np.arctan2(n,np.dot(a,b))).as_matrix()

class Person:
 def __init__(self, include_right=False):
  self.include_right=include_right
  self.right_hand=np.array([.51,.36,.925]) if include_right else None
  vertices=[];faces=[];group=''
  for line in (SOURCE/'base.obj').read_text().splitlines():
   if line.startswith('v '):vertices.append(list(map(float,line.split()[1:])))
   elif line.startswith('g '):group=line[2:]
   elif line.startswith('f ') and group=='body':
    ids=[int(x.split('/')[0])-1 for x in line.split()[1:]]
    for j in range(1,len(ids)-1):faces.append([ids[0],ids[j],ids[j+1]])
  self.vertices=np.asarray(vertices)*.1;self.faces=np.asarray(faces)
  self.rig=json.loads((SOURCE/'default.mhskel').read_text());self.weights=json.loads((SOURCE/'default_weights.mhw').read_text())['weights']
  self.bones=self.rig['bones'];self.head={n:self.vertices[self.rig['joints'][b['head']]].mean(0) for n,b in self.bones.items()}
  self.tail={n:self.vertices[self.rig['joints'][b['tail']]].mean(0) for n,b in self.bones.items()}
  self.names=list(self.bones);self.protected=[]
  self.rest_rotation=np.array([[0,0,-1],[-1,0,0],[0,1,0.]])
  self.rest_translation=np.array([.79,0,.65])-self.rest_rotation@self.head['root']

 def pose(self, hand, right_hand=None):
  transforms={};hand=np.asarray(hand)
  targets={'L':hand+np.array([.10,0,.016])}
  right_hand=self.right_hand if right_hand is None else right_hand
  if right_hand is not None:targets['R']=np.asarray(right_hand)+np.array([.10,0,.016])
  elbow=None
  def visit(n):
   nonlocal elbow
   if n in transforms:return transforms[n]
   p=self.bones[n]['parent'];r,t=visit(p) if p else (self.rest_rotation,self.rest_translation)
   h=self.head[n];world=r@h+t;direction=None
   if n=='spine02':r=Rotation.from_euler('y',-20,degrees=True).as_matrix()@r
   if n.startswith('upperleg01.'):direction=np.array([-.97,0,-.10])
   if n.startswith('lowerleg01.'):direction=np.array([.06,0,-1.])
   if n.startswith('foot.'):direction=np.array([-1.,0,0])
   if n=='upperarm01.R' and 'R' not in targets:direction=np.array([.15,0,-1.])
   if n=='lowerarm01.R' and 'R' not in targets:direction=np.array([-.35,-.1,-1.])
   side=n[-1]
   if n.startswith('upperarm01.') and side in targets:
    wrist_target=targets[side]
    l1=np.linalg.norm(self.head['lowerarm01.'+side]-h);l2=np.linalg.norm(self.head['wrist.'+side]-self.head['lowerarm01.'+side])
    delta=wrist_target-world;distance=np.linalg.norm(delta);u=delta/distance
    reach=min(distance,l1+l2-.001);a=(l1*l1-l2*l2+reach*reach)/(2*reach)
    perpendicular=np.array([0,-1.,0.]) if side=='L' else np.array([1.,.5,-.7]);perpendicular-=u*np.dot(perpendicular,u);perpendicular/=np.linalg.norm(perpendicular)
    elbow=world+u*a+perpendicular*np.sqrt(max(0,l1*l1-a*a));direction=elbow-world
    restdir=self.head['lowerarm01.'+side]-h
    r=between(r@restdir,direction)@r;direction=None
   if n.startswith('lowerarm01.') and side in targets:
    restdir=self.head['wrist.'+side]-h
    r=between(r@restdir,targets[side]-world)@r
   if n.startswith('wrist.') and side in targets:
    # Align metacarpal axis toward the robot and dorsal side upward.
    f=self.head['finger3-1.'+side]-h;f/=np.linalg.norm(f)
    across=self.head['finger5-1.'+side]-self.head['finger2-1.'+side];across-=f*np.dot(across,f);across/=np.linalg.norm(across)
    normal=np.cross(f,across)
    sign=1 if side=='L' else -1
    desired=np.column_stack(([-1.,0,0],[0,-sign,0],[0,0,sign]))
    r=desired@np.column_stack((f,across,normal)).T
   if n=='finger1-1.R' and 'R' in targets:
    # Thumb abduction keeps the thumb clear while fingertips contact the rim.
    r=Rotation.from_rotvec([-.55,0,0]).as_matrix()@r
   if direction is not None:r=between(r@(self.tail[n]-h),direction)@r
   t=world-r@h;transforms[n]=(r,t);return r,t
  for n in self.names:visit(n)
  return transforms

 def augment(self, xml, hand):
  self.protected=[]
  root=ET.fromstring(xml);asset=root.find('asset');world=root.find('worldbody');pose=self.pose(hand)
  # Research scenes carry their own redistributable texture source. The old
  # environment image files have incomplete per-asset attribution; none are
  # copied into our benchmark. Robot geometry stays under upstream MIT terms.
  wood=ROOT/'research/upstream/polyhaven/wood_table_diff_1k.png'
  if not wood.is_file():raise FileNotFoundError('Verified CC0 wood texture missing')
  for texture in asset.findall('texture'):
   if texture.get('file'):
    name=texture.get('name');texture.attrib.clear();texture.set('name',name);texture.set('type','2d')
    if 'table' in name: texture.set('file',str(wood))
    else:
     texture.set('builtin','checker' if 'floor' in name else 'flat');texture.set('width','64');texture.set('height','64')
     texture.set('rgb1','.43 .44 .45' if 'floor' in name else '.72 .73 .74');texture.set('rgb2','.47 .48 .49')
  for material in asset.findall('material'):
   if 'table' in material.get('name',''):
    material.set('reflectance','0');material.set('specular','.18');material.set('shininess','.15')
  # Disable obsolete standalone arm: body name is retained for BDDL registry.
  for geom in world.iter('geom'):
   if geom.get('name','').startswith('proxy_forearm_1_'):
    geom.set('rgba','0 0 0 0');geom.set('contype','0');geom.set('conaffinity','0')
  for body in world.iter('body'):
   if body.get('name','').startswith('proxy_forearm_1_'):body.set('gravcomp','1')
  for name in self.names:
   r,t=pose[name];h=self.head[name];pos=r@h+t
   body=ET.SubElement(world,'body',name='person_'+name,mocap='true',pos=fmt(pos),quat=fmt(quat(r)))
   radius=None
   active_side=name.endswith('.L') or (self.include_right and name.endswith('.R'))
   side=name[-1]
   if name.startswith('lowerarm') and active_side:radius=.040 if '01' in name else .029
   elif name.startswith('wrist.') and active_side:
    f=self.head['finger3-1.'+side]-h;f/=np.linalg.norm(f)
    across=self.head['finger5-1.'+side]-self.head['finger2-1.'+side];across-=f*np.dot(across,f);across/=np.linalg.norm(across)
    normal=np.cross(f,across);basis=np.column_stack((across,normal,f))
    geom_name='proxy_forearm_1_person_palm'+('_R' if side=='R' else '')
    ET.SubElement(body,'geom',name=geom_name,type='ellipsoid',pos=fmt((self.head['finger3-1.'+side]-h)*.5),
        quat=fmt(quat(basis)),size='.046 .025 .057',rgba='0 0 0 0',group='0')
    self.protected.append(geom_name)
   elif name.startswith(('finger','metacarpal')) and active_side:radius=.009 if '1-' not in name else (.012 if side=='R' else .011)
   if radius:
    delta=self.tail[name]-h
    geom_name='proxy_forearm_1_person_'+name
    offset=np.array([.00032 if side=='L' else -.00032,.00514,.00306]) if name.startswith('lowerarm01.') else np.zeros(3)
    ET.SubElement(body,'geom',name=geom_name,type='capsule',fromto=fmt([*offset,*(delta+offset)]),size=str(radius),rgba='0 0 0 0',group='0')
    self.protected.append(geom_name)
  # Independent CC0 garments, natural hair and textured eyes/skin.
  from .character_assets import load_character,render_arrays
  self.character_parts=load_character(self)
  for part in self.character_parts:
   points,blended,uv,faces=render_arrays(part);name='seated_person_'+part['name']
   ET.SubElement(asset,'texture',name=name+'_texture',type='2d',file=str(part['texture']))
   ET.SubElement(asset,'material',name=name+'_material',texture=name+'_texture',specular='.12',shininess='.12')
   skin=ET.SubElement(asset,'skin',name=name,vertex=fmt(points),face=fmt(faces),texcoord=fmt(uv),
       material=name+'_material',rgba='1 1 1 1',group='1')
   for j,bone in enumerate(self.names):
    indices=np.where(blended[:,j]>1e-6)[0]
    if not len(indices):continue
    ET.SubElement(skin,'bone',body='person_'+bone,bindpos=fmt(self.head[bone]),bindquat='1 0 0 0',
      vertid=' '.join(map(str,indices)),vertweight=fmt(blended[indices,j]))
  # A chair provides context and supports the seated pose visually.
  seat=ET.SubElement(world,'body',name='person_chair',pos='.80 0 .499')
  ET.SubElement(seat,'geom',name='person_chair_seat',type='box',size='.21 .22 .025',rgba='.16 .19 .22 1',contype='0',conaffinity='0',group='1')
  ET.SubElement(seat,'geom',name='person_chair_back',type='box',pos='.20 0 .24',size='.022 .22 .26',rgba='.16 .19 .22 1',contype='0',conaffinity='0',group='1')
  for x in [-.16,.16]:
   for y in [-.17,.17]:ET.SubElement(seat,'geom',type='cylinder',pos=fmt([x,y,-.27]),size='.015 .27',rgba='.23 .25 .26 1',contype='0',conaffinity='0',group='1')
  # Table is bench-height; a separate low footrest supports both feet rather
  # than leaving the seated avatar floating above the floor.
  rest=ET.SubElement(world,'body',name='person_footrest',pos='.24 0 .092')
  ET.SubElement(rest,'geom',name='person_footrest_top',type='box',size='.135 .255 .012',rgba='.13 .15 .17 1',contype='0',conaffinity='0',group='1')
  for y in [-.22,.22]:
   ET.SubElement(rest,'geom',type='box',pos=fmt([0,y,-.046]),size='.10 .012 .046',rgba='.20 .22 .24 1',contype='0',conaffinity='0',group='1')
  camera=world.find(".//camera[@name='agentview']")
  if camera is not None:
   position=np.array([.72,-.85,1.55]);target=np.array([.02,0,.96]);z=(position-target)/np.linalg.norm(position-target);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
   camera.set('pos',fmt(position));camera.set('quat',fmt(quat(np.column_stack([x,y,z]))))
  # Observer view includes the whole seated person; policy views remain actual sensors.
  pos=np.array([1.45,1.25,1.65]);target=np.array([.12,0,.97]);z=(pos-target)/np.linalg.norm(pos-target);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
  ET.SubElement(world,'camera',name='research_overview',pos=fmt(pos),quat=fmt(quat(np.column_stack([x,y,z]))),fovy='52')
  ET.SubElement(world,'camera',name='research_top',pos='0 0 2.15',quat='1 0 0 0',fovy='52')
  return ET.tostring(root,encoding='unicode')

 def update(self, sim, hand):
  pose=self.pose(hand)
  for n,(r,t) in pose.items():
   bid=sim.model.body_name2id('person_'+n);mid=sim.model.body_mocapid[bid]
   sim.data.mocap_pos[mid]=r@self.head[n]+t;sim.data.mocap_quat[mid]=quat(r)

 def surface_vertices(self, hand):
  """Asset QA only: CPU equivalent of the registered linear blend skinning."""
  posed=np.zeros_like(self.vertices);total=np.zeros(len(self.vertices));protected=np.zeros(len(self.vertices))
  pose=self.pose(hand)
  for n,entries in self.weights.items():
   if not entries:continue
   ids=np.array([i for i,w in entries]);weights=np.array([w for i,w in entries]);r,t=pose[n]
   posed[ids]+=(self.vertices[ids]@r.T+t)*weights[:,None];total[ids]+=weights
   if n.startswith(('lowerarm','finger','metacarpal','wrist.')) and (n.endswith('.L') or self.include_right and n.endswith('.R')):protected[ids]+=weights
  posed/=np.maximum(total[:,None],1e-10)
  used=np.unique(self.faces)
  return posed[used],used[protected[used]>.5],posed
