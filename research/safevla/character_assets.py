"""Original reader for CC0 MakeHuman mesh fitting data; no application code.

Separate garments/hair/eyes are skinned to the existing anatomical bones.
UV seams duplicate render vertices only. QA retains source mesh topology.
"""
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'research/upstream/makehuman-character'

def read_obj(path, group=None):
    vertices=[];uv=[];faces=[];current=''
    for line in path.read_text().splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':vertices.append(list(map(float,fields[1:4])))
        elif fields[0]=='vt':uv.append(list(map(float,fields[1:3])))
        elif fields[0]=='g':current=' '.join(fields[1:])
        elif fields[0]=='f' and (group is None or current==group):
            corners=[tuple(int(x)-1 for x in token.split('/')[:2]) for token in fields[1:]]
            for k in range(1,len(corners)-1):faces.append([corners[0],corners[k],corners[k+1]])
    return np.asarray(vertices)*.1,np.asarray(uv),np.asarray(faces)

def weight_matrix(person):
    matrix=np.zeros((len(person.vertices),len(person.names)))
    for j,name in enumerate(person.names):
        for vertex,weight in person.weights.get(name,[]):matrix[vertex,j]=weight
    return matrix/np.maximum(matrix.sum(1,keepdims=True),1e-12)

def load_part(person,relative):
    path=SOURCE/relative;lines=path.read_text().splitlines();metadata={};mapping=[];hidden=[];mode='header'
    for line in lines:
        t=line.split()
        if not t or t[0].startswith('#'):continue
        if t[0]=='verts':mode='verts';continue
        if t[0]=='delete_verts':mode='delete';continue
        if not t[0].lstrip('-').isdigit():metadata[t[0]]=t[1:];continue
        if mode=='delete':
            tokens=re.findall(r'\d+|\-',line);i=0
            while i<len(tokens):
                first=int(tokens[i]);i+=1
                if i<len(tokens) and tokens[i]=='-':hidden.extend(range(first,int(tokens[i+1])+1));i+=2
                else:hidden.append(first)
        elif mode=='verts':mapping.append(t)
    raw,uv,faces=read_obj(path.parent/metadata['obj_file'][0])
    if len(raw)!=len(mapping):raise ValueError('Mesh fitting count mismatch '+str(path))
    scale=np.ones(3)
    for axis,key in enumerate(['x_scale','y_scale','z_scale']):
        if key in metadata:
            a,b,d=metadata[key];scale[axis]=abs(person.vertices[int(a),axis]-person.vertices[int(b),axis])/(float(d)*.1)
    points=[];weights=[];base_weights=weight_matrix(person)
    for row in mapping:
        if len(row)==1:
            ids=[int(row[0])];blend=np.ones(1);offset=np.zeros(3)
        else:
            ids=list(map(int,row[:3]));blend=np.array(list(map(float,row[3:6])));offset=np.array(list(map(float,row[6:9])))*.1*scale
        points.append(blend@person.vertices[ids]+offset)
        w=np.maximum(0,blend@base_weights[ids]);weights.append(w/max(w.sum(),1e-10))
    material=path.parent/metadata['material'][0]
    diffuse=None
    for line in material.read_text().splitlines():
        if line.startswith('diffuseTexture '):diffuse=(material.parent/line.split(maxsplit=1)[1]).resolve()
    return {'name':path.stem,'points':np.array(points),'weights':np.array(weights),'uv':uv,'faces':faces,'hidden':hidden,'texture':diffuse}

def render_arrays(part):
    # MuJoCo skins share one UV index with each vertex; OBJ uses two indices.
    lookup={};ids=[];tex=[];tri=[]
    for face in part['faces']:
        row=[]
        for vi,ti in face:
            key=(int(vi),int(ti))
            if key not in lookup:lookup[key]=len(ids);ids.append(vi);tex.append(part['uv'][ti])
            row.append(lookup[key])
        tri.append(row)
    tex=np.asarray(tex);tex[:,1]=1-tex[:,1]
    return part['points'][ids],part['weights'][ids],tex,np.array(tri)

def load_character(person):
    parts=[load_part(person,p) for p in ['clothes/male_casualsuit06/male_casualsuit06.mhclo','hair/short04/short04.mhclo','eyes/high-poly/high-poly.mhclo']]
    points,uv,faces=read_obj(ROOT/'research/upstream/makehuman-assets/base.obj',group='body')
    hidden=set(i for p in parts for i in p['hidden'])
    visible=[f for f in faces if not any(int(vi) in hidden for vi,ti in f)]
    body={'name':'body','points':person.vertices,'weights':weight_matrix(person),'uv':uv,'faces':np.asarray(visible),'hidden':[],
          'texture':SOURCE/'skins/young_asian_male/young_lightskinned_male_diffuse3.png'}
    return [body,*parts]

def posed_part(part,person,pose):
    result=np.zeros_like(part['points'])
    for j,name in enumerate(person.names):
        r,t=pose[name];result+=(part['points']@r.T+t)*part['weights'][:,j,None]
    return result
