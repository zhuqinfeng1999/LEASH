"""Paired source/device slot translation plus a real adjacent cart."""
from dataclasses import replace
from unittest.mock import patch
from xml.etree import ElementTree as E
from research.v4.accelerated import tabletop
from research.v4.candidates.user_rework_r3_20260927.scene import FrontFixtureEnv
from research.v4.candidates.user_rework_r3_20260927.soup_low_table_v8.scene import SoupLowTableScene
from research.v4.candidates.r3_02_horizontal_can_cradle_readonly_20260928 import scene as geometry
from .config import XY,OFFSET,OVEN_OFFSET,CART_X,CART_Y,CART_TOP,CART_THICK

class CartFixtureEnv(FrontFixtureEnv):
    def _load_model(self):
        geometry.CradleFixtureEnv._load_model(self)
        p,_=tabletop.sample_slot('L',self.layout_seed,object_index=0)
        p[:2]+=OFFSET;p[2]=.9
        cradle=self.model.worldbody.find("body[@name='r302_real_wood_v_cradle']")
        if cradle is None:raise ValueError('Real cradle seam changed')
        cradle.set('pos',' '.join(map(str,p)))
        cx=(CART_X[0]+CART_X[1])/2;cy=(CART_Y[0]+CART_Y[1])/2
        b=E.SubElement(self.model.worldbody,'body',name='r302_plain_appliance_cart',pos='0 0 0')
        E.SubElement(b,'geom',name='r302_cart_top',type='box',pos=f'{cx} {cy} {CART_TOP-CART_THICK/2}',size=f'{(CART_X[1]-CART_X[0])/2} {(CART_Y[1]-CART_Y[0])/2} {CART_THICK/2}',rgba='.72 .69 .63 1',contype='1',conaffinity='1',friction='1 .005 .0001')
        bottom=CART_TOP-CART_THICK
        for i,x in enumerate((CART_X[0]+.04,CART_X[1]-.04)):
            for j,y in enumerate((CART_Y[0]+.04,CART_Y[1]-.04)):
                E.SubElement(b,'geom',name=f'r302_cart_leg_{i}_{j}',type='box',pos=f'{x} {y} {bottom/2}',size=f'.015 .015 {bottom/2}',rgba='.20 .22 .24 1',contype='1',conaffinity='1')
                # Horizontal physical foot atfloor, not a floating/disabled leg.
                E.SubElement(b,'geom',name=f'r302_cart_foot_{i}_{j}',type='box',pos=f'{x} {y} .006',size='.026 .026 .006',rgba='.09 .09 .09 1',contype='1',conaffinity='1')

class CartRelayoutScene(SoupLowTableScene):
    layout_version='R302-paired-source-device-X-translation-real-floor-cart-v1'
    def __init__(self,*args,**kwargs):
        original=tabletop.ObjectBinding
        def binding(*a,**k):
            v=original(*a,**k)
            if v.name==self.item_name:
                if v.xy_offset!=(-.14,0):raise ValueError('Parent can seam changed')
                v=replace(v,xml=str(geometry.H/'assets/horizontal_soup_can.xml'),z_offset=geometry.CENTER_Z,grasp_local=(-.035,0.,0.),xy_offset=OFFSET)
            elif v.name==self.oven_name:
                if v.xy_offset!=(.10,-.24):raise ValueError('Parent oven seam changed')
                v=replace(v,xy_offset=OVEN_OFFSET)
            return v
        with patch.object(tabletop,'ObjectBinding',binding),patch.object(FrontFixtureEnv,'_load_model',CartFixtureEnv._load_model):
            super().__init__(*args,**kwargs)
