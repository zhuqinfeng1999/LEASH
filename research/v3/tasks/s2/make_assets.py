"""Create original, plain MuJoCo geometry for the S2 mug, dish, and pills.

No third-party mesh, texture, text, or branding is embedded here.
"""
from pathlib import Path
from math import cos, pi, sin
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent


def _base(name):
    root = ET.Element('mujoco', model=name)
    asset = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    outer = ET.SubElement(world, 'body')
    body = ET.SubElement(outer, 'body', name='object')
    return root, asset, outer, body


def _write(name, root):
    ET.indent(root, space='  ')
    (HERE / f'{name}.xml').write_text(ET.tostring(root, encoding='unicode') + '\n')


def mug():
    root, asset, outer, body = _base('s2_milk_mug')
    ET.SubElement(asset, 'material', name='ceramic', rgba='.16 .31 .47 1', specular='.48', shininess='.5')
    ET.SubElement(asset, 'material', name='milk', rgba='.96 .95 .87 1', specular='.28', shininess='.22')
    ET.SubElement(body, 'inertial', pos='0 0 .044', mass='.32', diaginertia='.00038 .00038 .00045')
    ET.SubElement(body, 'geom', name='base', type='cylinder', pos='0 0 .004', size='.045 .004', material='ceramic', group='0', friction='.85 .005 .001')
    n = 24
    for i in range(n):
        a = 2 * pi * i / n
        attrs = dict(name=f'wall_{i:02d}', type='box', pos=f'{.042*cos(a):.7f} {.042*sin(a):.7f} .047',
                     size='.0048 .0065 .043', quat=f'{cos(a/2):.8f} 0 0 {sin(a/2):.8f}',
                     material='ceramic', group='0', friction='.75 .005 .001')
        ET.SubElement(body, 'geom', **attrs)
    handle_points = ((.044, .018), (.071, .020), (.080, .044),
                     (.071, .069), (.044, .072))
    for i, (a, b) in enumerate(zip(handle_points, handle_points[1:])):
        ET.SubElement(body, 'geom', name=f'handle_{i}', type='capsule',
                      fromto=f'{a[0]} 0 {a[1]} {b[0]} 0 {b[1]}', size='.006',
                      material='ceramic', group='0')
    ET.SubElement(body, 'geom', name='milk_surface', type='cylinder', pos='0 0 .076', size='.037 .0006',
                  material='milk', group='1', contype='0', conaffinity='0')
    ET.SubElement(body, 'site', name='bottom_site', pos='0 0 0', size='.001')
    ET.SubElement(body, 'site', name='top_site', pos='0 0 .09', size='.001')
    ET.SubElement(body, 'site', name='horizontal_radius_site', pos='.082 0 0', size='.001')
    _write('milk_mug', root)


def dish():
    root, asset, outer, body = _base('s2_pill_dish')
    ET.SubElement(asset, 'material', name='dish_ceramic', rgba='.91 .89 .83 1', specular='.35', shininess='.28')
    ET.SubElement(body, 'inertial', pos='0 0 .008', mass='.23', diaginertia='.0016 .0016 .0025')
    ET.SubElement(body, 'geom', name='dish_base', type='cylinder', pos='0 0 .004', size='.135 .004',
                  material='dish_ceramic', group='0', friction='.8 .005 .001')
    n = 32
    for i in range(n):
        a = 2 * pi * i / n
        ET.SubElement(body, 'geom', name=f'rim_{i:02d}', type='box',
                      pos=f'{.13*cos(a):.7f} {.13*sin(a):.7f} .014',
                      size='.004 .013 .012', quat=f'{cos(a/2):.8f} 0 0 {sin(a/2):.8f}',
                      material='dish_ceramic', group='0', friction='.8 .005 .001')
    ET.SubElement(body, 'site', name='bottom_site', pos='0 0 0', size='.001')
    ET.SubElement(body, 'site', name='top_site', pos='0 0 .026', size='.001')
    ET.SubElement(body, 'site', name='horizontal_radius_site', pos='.135 0 0', size='.001')
    _write('pill_dish', root)


def pill():
    root, asset, outer, body = _base('s2_sleeping_pill')
    ET.SubElement(asset, 'material', name='ivory_pill', rgba='.91 .92 .85 1', specular='.24', shininess='.2')
    ET.SubElement(body, 'inertial', pos='0 0 0', mass='.018', diaginertia='.000006 .00002 .00002')
    ET.SubElement(body, 'geom', name='pill_body', type='capsule', fromto='-.020 0 0 .020 0 0', size='.014',
                  material='ivory_pill', group='0', friction='1.2 .005 .001')
    ET.SubElement(body, 'geom', name='score_line', type='box', pos='0 0 .014', size='.0007 .008 .0004',
                  rgba='.67 .68 .63 1', group='1', contype='0', conaffinity='0')
    ET.SubElement(body, 'site', name='bottom_site', pos='0 0 -.014', size='.001')
    ET.SubElement(body, 'site', name='top_site', pos='0 0 .014', size='.001')
    ET.SubElement(body, 'site', name='horizontal_radius_site', pos='.034 0 0', size='.001')
    _write('sleeping_pill', root)


if __name__ == '__main__':
    mug()
    dish()
    pill()
