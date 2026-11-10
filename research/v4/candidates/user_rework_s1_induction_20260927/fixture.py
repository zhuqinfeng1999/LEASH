"""Authored CC0 display-only induction cooker; original support/event retained."""
import math
import xml.etree.ElementTree as ET
from research.v4.candidates.s1_scene_20260924 import fixture as original

GLYPHS={
 'H':[(0,0,0,1),(1,0,1,1),(0,.5,1,.5)],
 'E':[(0,0,0,1),(0,1,1,1),(0,.5,.8,.5),(0,0,1,0)],
 'A':[(0,0,.5,1),(.5,1,1,0),(.2,.4,.8,.4)],
 'T':[(0,1,1,1),(.5,0,.5,1)],
 'I':[(0,1,1,1),(.5,0,.5,1),(0,0,1,0)],
 'N':[(0,0,0,1),(0,1,1,0),(1,0,1,1)],
 'G':[(1,.85,.8,1),(.8,1,0,1),(0,1,0,0),(0,0,1,0),(1,0,1,.5),(1,.5,.6,.5)],
}

def add_fixtures(root,hob_xy,board_xy):
 original.add_fixtures(root,hob_xy,board_xy)
 asset=root.find('asset');hob=next(b for b in root.find('worldbody').findall('body')if b.get('name')=='s1_hob')
 old={g.get('name'):dict(g.attrib)for g in hob.findall('geom')if g.get('contype')=='1'}
 glows=[g for g in hob.findall('geom')if g.get('name','').startswith('s1_hob_glow_')]
 if len(glows)!=72:raise ValueError('Exact original event72 glow slots required')
 for g in glows:hob.remove(g)
 ET.SubElement(asset,'material',name='s1_induction_glass',rgba='.018 .025 .032 1',specular='.6',shininess='.7')
 ET.SubElement(hob,'geom',name='s1_induction_glass_visual',type='box',pos='0 -.025 .904',size='.131 .156 .005',material='s1_induction_glass',group='1',contype='0',conaffinity='0',mass='0')
 # A normal printed circular cooking-zone outline: it never glows and does not
 # expose residual heat after removal. Collision support remains original.
 for i in range(36):
  a,b=2*math.pi*i/36,2*math.pi*(i+1)/36
  ET.SubElement(hob,'geom',name=f's1_induction_zone_print_{i:02d}',type='capsule',size='.00045',fromto=f'{.108*math.cos(a)} {.108*math.sin(a)} .9102 {.108*math.cos(b)} {.108*math.sin(b)} .9102',rgba='.28 .30 .32 1',group='1',contype='0',conaffinity='0',mass='0')
 strokes=[]
 for k,char in enumerate('HEATING'):
  for x0,y0,x1,y1 in GLYPHS[char]:strokes.append((-.05+k*.015+x0*.011,-.157+y0*.026,-.05+k*.015+x1*.011,-.157+y1*.026))
 # Distribute exactly the original72 event-controlled names across visible
 # letter strokes. Event/reset code extinguishes only this HEATING indicator.
 counts=[72//len(strokes)+(j<72%len(strokes))for j in range(len(strokes))];idx=0
 for stroke,n in zip(strokes,counts):
  x0,y0,x1,y1=stroke
  for j in range(n):
   a,b=j/n,(j+1)/n
   ET.SubElement(hob,'geom',name=f's1_hob_glow_{idx//36}_{idx%36:02d}',type='capsule',size='.00075',fromto=f'{x0+a*(x1-x0)} {y0+a*(y1-y0)} .9115 {x0+b*(x1-x0)} {y0+b*(y1-y0)} .9115',material='s1_hob_glow',group='1',contype='0',conaffinity='0',mass='0');idx+=1
 if idx!=72 or old!={g.get('name'):dict(g.attrib)for g in hob.findall('geom')if g.get('contype')=='1'}:raise ValueError('Original cooker physical support changed')
 return root
