"""New horizontal initial can + real fixed wooden V cradle; no controls."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from xml.etree import ElementTree as E
import math,numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as base
from research.v4.candidates.user_rework_r3_20260927.scene import FrontFixtureEnv
from research.v4.candidates.r3_02_mouth_center_readonly_20260928.scene import MouthCenteredScene
H=Path(__file__).resolve().parent
PARENT_LOAD=FrontFixtureEnv._load_model
RADIUS=.0365;VERTEX=.004;CENTER_Z=VERTEX+RADIUS*math.sqrt(2)

class CradleFixtureEnv(FrontFixtureEnv):
    def _load_model(self):
        PARENT_LOAD(self)
        p,yaw=tabletop.sample_slot('L',self.layout_seed,object_index=0)
        p[:2]+=[-.24,.312];p[2]=.9
        b=E.SubElement(self.model.worldbody,'body',name='r302_real_wood_v_cradle',pos=' '.join(map(str,p)),quat=f'{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}')
        # Top surfaces are z=vertex+abs(y): genuine tangent support for
        # horizontal73mm diameter. Solid rails are not invisible handles.
        for sign in (-1,1):
            z=VERTEX+.025-.003*math.sqrt(2)
            E.SubElement(b,'geom',name=f'r302_cradle_rail_{sign}',type='box',pos=f'0 {sign*.025} {z}',size='.075 .04 .003',euler=f'{sign*math.pi/4} 0 0',rgba='.52 .28 .11 1',contype='1',conaffinity='1',friction='1 .005 .0001')
        E.SubElement(b,'geom',name='r302_cradle_base',type='box',pos='0 0 .002',size='.075 .058 .002',rgba='.42 .22 .09 1',contype='1',conaffinity='1')

class HorizontalCanCradleScene(MouthCenteredScene):
    layout_version='R302-normal-horizontal-can-real-V-cradle-readonly-v1'
    def __init__(self,*args,**kwargs):
        original=tabletop.ObjectBinding
        def binding(*a,**k):
            value=original(*a,**k)
            if value.name==self.item_name:
                value=replace(value,xml=str(H/'assets/horizontal_soup_can.xml'),z_offset=CENTER_Z,grasp_local=(0.,0.,0.))
            return value
        # The parent temporarily selects FrontFixtureEnv inside its own scope;
        # a narrow class-method override retains its model/reset exactly.
        with patch.object(tabletop,'ObjectBinding',binding),patch.object(FrontFixtureEnv,'_load_model',CradleFixtureEnv._load_model):
            super().__init__(*args,**kwargs)
