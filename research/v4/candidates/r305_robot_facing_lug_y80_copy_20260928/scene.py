"""Unrun normal constructor proposal; no episode-time pose/control writes."""
from types import FunctionType
from unittest.mock import patch
import numpy as np

from research.v4.accelerated import tabletop
from research.v4.candidates.r305_tall_microwave_inside12_20260928 import scene as parent
from .reset_orientation import ORIGINAL_SAMPLE, YAW_OFFSET

ORIGINAL_LOAD=parent.ORIGINAL_LOADER
ORIGINAL_INIT=parent.TallLugScene.__init__


class RobotFacingLugY80Scene(parent.TallLugScene):
    layout_version='R305-supported-device-cart-Y80-robot-facing-positive-ear-side-grip-v1'

    def __init__(self,*args,**kwargs):
        if parent.ENV._load_model is not ORIGINAL_LOAD or tabletop.sample_slot is not ORIGINAL_SAMPLE:
            raise RuntimeError('Original own normal constructor seam already changed')
        receipts=[]
        def sample(slot,seed,*,object_index=0):
            xyz,yaw=ORIGINAL_SAMPLE(slot,seed,object_index=object_index)
            if object_index==0:
                if slot!='L':raise RuntimeError('Own bowl source reset slot differs')
                yaw+=YAW_OFFSET
            elif object_index==1:
                if slot!='R':raise RuntimeError('Own microwave reset slot differs')
                xyz[1]+=.080
            elif object_index!=2 or slot!='L':
                raise RuntimeError('Own original three-binding reset order differs')
            receipts.append(dict(object_index=object_index,slot=slot,xyz=xyz.tolist(),yaw=yaw))
            return xyz,yaw
        def load(env):
            ORIGINAL_LOAD(env)
            carts=[e for e in env.model.root.iter('body')if e.get('name')=='r302_plain_appliance_cart']
            if len(carts)!=1:raise RuntimeError('Exact visible supported cart compile seam differs')
            old=np.fromstring(carts[0].get('pos','0 0 0'),sep=' ')
            if not np.array_equal(old,[0.,0.,0.]):raise RuntimeError('Original cart pose differs')
            carts[0].set('pos','0 .080 0')
        # Rebind the existing qualified TallLugScene initializer's one loader
        # reference, rather than copy its reset/controller/20-neutral executor.
        ns=dict(ORIGINAL_INIT.__globals__);ns['ORIGINAL_LOADER']=load
        init=FunctionType(ORIGINAL_INIT.__code__,ns,ORIGINAL_INIT.__name__,
                          ORIGINAL_INIT.__defaults__,ORIGINAL_INIT.__closure__)
        with patch.object(parent.ENV,'_load_model',load),patch.object(tabletop,'sample_slot',sample):
            init(self,*args,**kwargs)
        if parent.ENV._load_model is not ORIGINAL_LOAD or tabletop.sample_slot is not ORIGINAL_SAMPLE:
            raise RuntimeError('Own constructor source scopes were not restored')
        if not receipts or set(r['object_index']for r in receipts)!={0,1,2}:
            self.close();raise RuntimeError('Actual normal reset consumers absent')
        self.robot_facing_reset_receipts=receipts
        self.robot_facing_constructor_scope_restored=True
        # common_reset_controls=20 and original sequence/scorer remain inherited.
