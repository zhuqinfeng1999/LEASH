"""Physical/visual central short V; original can, region, oven and robot."""
from unittest.mock import patch
from research.v4.candidates.r302_rear_presentation_v1_20260928 import scene as rear
from .layout import HALF_LENGTH


class CentralShortVEnv(rear.TurnedVCradleCartEnv):
    def _load_model(self):
        super()._load_model()
        body=self.model.worldbody.find(".//body[@name='r302_v_wood_cradle']")
        if body is None: raise RuntimeError('Own original real V body missing')
        for name in ('r302_v_rail_-1','r302_v_rail_1','r302_v_base'):
            geom=body.find(f"geom[@name='{name}']")
            if geom is None: raise RuntimeError('Own real V geom missing: '+name)
            size=list(map(float,geom.get('size').split()))
            if size[0]!=.075: raise RuntimeError('Own original V length changed')
            size[0]=HALF_LENGTH
            geom.set('size',' '.join(f'{x:.17g}' for x in size))


class ShortVHorizontalCanScene(rear.RearPresentedCanScene):
    layout_version='R302-horizontal-original-plus42-central-short-realV-v1'
    def __init__(self,*args,**kwargs):
        with patch.object(rear,'TurnedVCradleCartEnv',CentralShortVEnv):
            super().__init__(*args,**kwargs)
