"""Original display-only electric glow and project-authored board fixture."""
from __future__ import annotations
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from .hob import HOB_XY,BOARD_XY,HOB_TOP_Z_M

HERE=Path(__file__).resolve().parent
BOARD_TEXTURE=HERE/'wood_board_1k.png'
RING_SEGS=36
RING_RADII_M=(.115,.121)


def make_texture():
    n=1024
    y,x=np.mgrid[0:n,0:n].astype(np.float32)
    grain=8*np.sin(.062*y+1.8*np.sin(.015*x))+3*np.sin(.31*y+.009*x)
    noise=np.random.default_rng(101).normal(0,1.8,(n,n)).astype(np.float32)
    value=(grain+noise)[...,None]
    rgb=np.clip(np.array([165,119,77],dtype=np.float32)+value,0,255).astype(np.uint8)
    Image.fromarray(rgb).save(BOARD_TEXTURE)


def add_fixtures(root:ET.Element,hob_xy=HOB_XY,board_xy=BOARD_XY):
    asset=root.find('asset')
    if asset is None:asset=ET.Element('asset');root.insert(0,asset)
    ET.SubElement(asset,'texture',name='s1_board_wood_tex',type='2d',file=str(BOARD_TEXTURE))
    ET.SubElement(asset,'material',name='s1_board_wood',texture='s1_board_wood_tex',
                  rgba='1 1 1 1',specular='.14',shininess='.13')
    ET.SubElement(asset,'material',name='s1_hob_dark',rgba='.075 .085 .095 1',
                  specular='.55',shininess='.53')
    ET.SubElement(asset,'material',name='s1_hob_glow',rgba='1 .075 .015 1',
                  emission='.9',specular='.15')
    world=root.find('worldbody')
    if world is None:raise ValueError('worldbody missing')
    hob=ET.SubElement(world,'body',name='s1_hob',pos=f'{hob_xy[0]} {hob_xy[1]} 0')
    ET.SubElement(hob,'geom',name='s1_hob_plate_collision',type='cylinder',
                  pos='0 0 .904',size=f'.112 {HOB_TOP_Z_M-.904:.6f}',
                  material='s1_hob_dark',group='0',contype='1',conaffinity='1')
    # Both concentric glowing bands remain outside the 103 mm pan body radius.
    for ring,radius in enumerate(RING_RADII_M):
        for segment in range(RING_SEGS):
            a=2*math.pi*segment/RING_SEGS
            b=2*math.pi*(segment+1)/RING_SEGS
            ET.SubElement(hob,'geom',name=f's1_hob_glow_{ring}_{segment:02d}',
                          type='capsule',size='.0027',
                          fromto=f'{radius*math.cos(a):.7f} {radius*math.sin(a):.7f} .912 '
                                 f'{radius*math.cos(b):.7f} {radius*math.sin(b):.7f} .912',
                          material='s1_hob_glow',group='1',contype='0',conaffinity='0',mass='0')
    board=ET.SubElement(world,'body',name='s1_board',pos=f'{board_xy[0]} {board_xy[1]} 0')
    ET.SubElement(board,'geom',name='s1_board_collision',type='box',
                  pos='0 0 .904',size='.135 .130 .006',material='s1_board_wood',
                  group='0',contype='1',conaffinity='1')
    # The board top is at the same 0.910 m height as the hob support.
    return root


def standalone_fixture_xml():
    root=ET.Element('mujoco',model='s1_fixture_cpu')
    ET.SubElement(root,'asset')
    ET.SubElement(root,'worldbody')
    add_fixtures(root)
    return ET.tostring(root,encoding='unicode')
