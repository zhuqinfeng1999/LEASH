"""One source+real-V slot parameter, applied in original normal initialization."""
import copy
import json
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r3_03_t1_derived_20260926 import scene as source_config
from research.v4.candidates.r302_horizontal_short_v_20260928 import scene as wood_select
from research.v4.candidates.r302_short_v_raised_neutral_20260928 import scene as raised

H=Path(__file__).resolve().parent
FRONT_SOURCE_Y=-.312266002685539
ORIGINAL_SOURCE_Y=-.16
DY=FRONT_SOURCE_Y-ORIGINAL_SOURCE_Y
ORIGINAL_CONFIGURATION=source_config.configuration


def front_configuration():
    c=copy.deepcopy(ORIGINAL_CONFIGURATION())
    if c['id']!='R3-02' or c['layout']['source_dy_m']!=.08:
        raise RuntimeError('Exact own can source configuration changed')
    c['layout']['source_dy_m']+=DY
    c['layout']['initial_pose']='original canonical raised-neutral PARK; front-source slot'
    return c


class FrontShortVEnv(wood_select.CentralShortVEnv):
    def _load_model(self):
        super()._load_model()
        body=self.model.worldbody.find(".//body[@name='r302_v_wood_cradle']")
        if body is None:raise RuntimeError('Actual central real V XML body missing')
        p=np.asarray(list(map(float,body.get('pos').split())))
        expected,_=tabletop.sample_slot('L',self.layout_seed,object_index=0)
        expected[:2]+=np.asarray((0,.08));expected[2]=tabletop.TABLE_Z
        if not np.array_equal(p,expected):raise RuntimeError('Actual original source/V placement seam changed')
        p[1]+=DY
        body.set('pos',' '.join(f'{x:.17g}' for x in p))


_RAISED_INIT=FunctionType(raised.RaisedNeutralShortVCanScene.__init__.__code__,
                         dict(raised.RaisedNeutralShortVCanScene.__init__.__globals__,H=H),
                         name='original_raised_init',
                         closure=raised.RaisedNeutralShortVCanScene.__init__.__closure__)


class FrontSlotShortVCanScene(raised.RaisedNeutralShortVCanScene):
    layout_version='R302-front-family-source-slot-short-realV-canonical-raised-horizontal-v1'
    def __init__(self,*args,**kwargs):
        old_cfg=source_config.configuration;old_wood=wood_select.CentralShortVEnv
        with patch.object(source_config,'configuration',front_configuration),patch.object(wood_select,'CentralShortVEnv',FrontShortVEnv):
            _RAISED_INIT(self,*args,**kwargs)
        m,d=self.backend.sim.model._model,self.backend.sim.data._data
        can=m.body('steel_thermos_1_main').id;wood=m.body('r302_v_wood_cradle').id
        expected,_=tabletop.sample_slot('L',self.backend.layout_seed,object_index=0)
        expected[:2]+=np.asarray((0,.08+DY));expected[2]=.9
        cp=d.xpos[can].copy();wp=d.xpos[wood].copy()
        passed=bool(np.allclose(wp,expected,rtol=0,atol=1e-15)
                    and np.allclose(cp[:2],wp[:2],rtol=0,atol=1e-15)
                    and source_config.configuration is old_cfg and wood_select.CentralShortVEnv is old_wood)
        report={'kind':'actual coupled source reset/config + precompile real V XML placement',
          'passed':passed,'nominal_source_xy_m':[-.08,FRONT_SOURCE_Y],
          'unique_dy_m':DY,'seed':self.backend.layout_seed,'actual_initial_can_center_m':cp.tolist(),
          'actual_initial_V_body_m':wp.tolist(),'expected_jittered_source_V_m':expected.tolist(),
          'original_XY_yaw_randomization_unchanged':True,'source_V_XY_relative_m':(cp-wp)[:2].tolist(),
          'configuration_scope_restored':source_config.configuration is old_cfg,
          'wood_env_scope_restored':wood_select.CentralShortVEnv is old_wood,
          'new_wrapper_live_qpos_writes':0,'source_sequence':self.sequence,
          'V_geom_group_native_visibility_pending':{n:int(m.geom_group[m.geom(n).id]) for n in ('r302_v_rail_-1','r302_v_rail_1','r302_v_base')},
          'not_old_can_script_qualification':True}
        (H/'initial_source_placement.json').write_text(json.dumps(report,indent=2)+'\n')
        if not passed:raise RuntimeError('Coupled actual source/V initialization mismatch')
