"""Only visible/colliding wood shape; can/robot/appliance contact unchanged."""
from xml.etree import ElementTree as ET
import numpy as np
from .layout import LAYOUT, WOOD_NAMES


def vector(values):return ' '.join(format(float(v),'.17g') for v in values)


def apply(root):
    body=root.find(".//body[@name='r302_v_wood_cradle']")
    if body is None:raise RuntimeError('Exact own source support body missing')
    old={n:body.find("geom[@name='"+n+"']") for n in WOOD_NAMES[:3]}
    if any(v is None for v in old.values()) or len(body.findall('geom'))!=3:
        raise RuntimeError('Expected exactly original central V three solids')
    if any(body.find("geom[@name='"+n+"']") is not None for n in WOOD_NAMES[3:]):
        raise RuntimeError('Flat-ledge transformation is one explicit version, not repeatable')
    p=root.find('contact'); pairs=[] if p is None else list(p.findall('pair'))
    if len(pairs)!=2 or {x.get('name')for x in pairs}!={'r302_source_pair_-1','r302_source_pair_1'}:
        raise RuntimeError('Exact old two V-pair identity changed')
    for x in pairs:
        if x.get('geom1')!='steel_thermos_1_can_body_collision' or x.get('geom2') not in WOOD_NAMES[1:3]:
            raise RuntimeError('Unexpected contact pair cannot be removed')
        p.remove(x)
    z=LAYOUT['floor_top_local_m']; a,b=LAYOUT['inner_half_footprint_m']; ao,bo=LAYOUT['outer_half_footprint_m']
    old['r302_v_base'].set('size',vector([ao,bo,z/2]));old['r302_v_base'].set('pos',vector([0,0,z/2]))
    for sign in (-1,1):
        node=old['r302_v_rail_'+str(sign)]
        node.set('size',vector([a,.0015,.003]));node.set('pos',vector([0,sign*(b+.0015),z+.003]))
        node.set('quat','1 0 0 0');node.attrib.pop('euler',None)
        end=ET.SubElement(body,'geom',name='r302_ledge_end_'+str(sign),type='box',
            size=vector([.0015,b,.003]),pos=vector([sign*(a+.0015),0,z+.003]),
            rgba='.52 .28 .11 1',contype='1',conaffinity='1',friction='1 .005 .0001',group='2')
    for node in old.values():
        node.set('group','2')
    return dict(layout=LAYOUT,removed_exact_pairs=[x.get('name')for x in pairs],
        can_collision_parameters_unchanged=True,new_support_not_physical_equivalent=True,
        original_masks_friction_solref_default_unchanged=True,
        visible_real_wood_geoms=list(WOOD_NAMES),new_adhesion_or_weld=False)
