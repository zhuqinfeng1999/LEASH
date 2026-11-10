"""ONE own source layout: actual60mm wood plinth, raised own foil+board.
Normal configuration/reset only; scoped proposed mirrored initial joint7.
"""
import copy,math
from unittest.mock import patch
from xml.etree import ElementTree as E
from research.v4.candidates.r3_04_horizontal_side_grasp_copy_20260928 import scene as donor
from research.v4.candidates.r3_04_foil_sandwich_20260926 import scene as own
from research.v4.accelerated import tabletop
OLD_Q=tuple(own.NEUTRAL_QPOS)
NEW_Q=OLD_Q[:-1]+(OLD_Q[-1]+math.pi,)
RISE=.060
# Whole original board/source XY±20mm, yaw±10deg; no added binding/index jitter.
MAX_YAW=math.radians(10)
BOARD_HALF=(.110,.090)
PLINTH_CENTER=(tabletop.SLOTS['L'].xyz[0],tabletop.SLOTS['L'].xyz[1]+.08)
PLINTH_HALF=(.110*math.cos(MAX_YAW)+.090*math.sin(MAX_YAW)+.020+.005,
             .110*math.sin(MAX_YAW)+.090*math.cos(MAX_YAW)+.020+.005)
ORIGINAL_CONFIG=donor.configuration

def configuration():
    c=copy.deepcopy(ORIGINAL_CONFIG())
    if c['id']!='R3-04' or c['metal_item']['z_offset']!=.039 or c['support']['z_offset']!=.009 or c['layout']['source_dy_m']!=.08:raise RuntimeError('Exact own source/board normal reset configuration changed')
    c['metal_item']['z_offset']+=RISE;c['support']['z_offset']+=RISE
    return c

class SourcePlinthCartEnv(donor.OwnCartEnv):
    def _load_model(self):
        super()._load_model()
        if self.model.worldbody.find("body[@name='r304_fixed_wood_source_plinth60']")is not None:raise RuntimeError('Duplicate source plinth')
        if self.model.asset.find("material[@name='sandwich_board_1_wood']")is None:raise RuntimeError('Exact original board wood material not available')
        x,y=PLINTH_CENTER;hx,hy=PLINTH_HALF
        b=E.SubElement(self.model.worldbody,'body',name='r304_fixed_wood_source_plinth60',pos=f'{x} {y} {.9+RISE/2}')
        E.SubElement(b,'geom',name='r304_source_plinth60_collision',type='box',size=f'{hx} {hy} {RISE/2}',material='sandwich_board_1_wood',group='0',mass='0',contype='1',conaffinity='1',friction='1 .005 .0001',solref='.02 1')
        E.SubElement(b,'geom',name='r304_source_plinth60_visual',type='box',size=f'{hx} {hy} {RISE/2}',material='sandwich_board_1_wood',group='1',mass='0',contype='0',conaffinity='0')

class RaisedSideFoilScene(donor.HorizontalSideFoilScene):
    layout_version='R304-real-source-board-wood-plinth60-mirrored-initial-v1'
    def __init__(self,*args,**kwargs):
        if tuple(own.NEUTRAL_QPOS)!=OLD_Q:raise RuntimeError('Original own initial seam changed')
        with patch.object(donor,'configuration',configuration),patch.object(donor,'OwnCartEnv',SourcePlinthCartEnv),patch.object(own,'NEUTRAL_QPOS',NEW_Q):
            super().__init__(*args,**kwargs)
