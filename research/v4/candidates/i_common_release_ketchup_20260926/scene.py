"""One common original bottle replacement on the qualified I1 v7/v9 scene.

All non-bottle bindings, natural arms, paper pixels, fixed microwave, cameras,
controller and hazard rules stay inherited. No task/name/pose API goes online.
"""
from dataclasses import replace
from pathlib import Path
from research.v4.accelerated import tabletop
from research.v4.candidates.i1_natural_corridor_20260925 import scene as original
from research.v4.candidates.i1_straight_right_arm_20260925.scene import I1StraightRightEnv
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,KETCHUP_REGIONS
ROOT=Path(__file__).resolve().parents[4]
BOTTLE='research/v4/candidates/release_ketchup_20260926/ketchup.xml'
LEFT_HAND=original.LEFT_HAND;RIGHT_HAND=original.RIGHT_HAND
PROTOCOL='i1-common-original-cc0-ketchup-authored-body-neck-region-v1'

def replace_bottle_bindings(bindings):
    """Reusable I2/I3/I4 configuration hook, not their automatic migration."""
    values=tuple(bindings)
    if sum(b.name=='ketchup_1'for b in values)!=1:raise ValueError('requires exactly one existing ketchup_1 binding')
    return tuple(replace(b,xml=BOTTLE,z_offset=0.,grasp_local=(0.,0.,.116))if b.name=='ketchup_1'else b for b in values)
BINDINGS=replace_bottle_bindings(original.BINDINGS)

def make_assist(env):
    body=env.objects_dict['ketchup_1'].root_body
    return AuthoredRegionGraspAssist(env.sim.model._model,env.sim.data._data,
        authored_regions={body:KETCHUP_REGIONS},**env.grasp_assist_kwargs)

def create(seed,*,camera_observations=False,render_gpu_device_id=-1):
    # The existing thin carrier invokes person/fixed-prop loading on E1 keys.
    # Keep its original private key so microwave/note/person stay present.
    key='E1-01'
    existing=tabletop.PILOTS.get(key);tabletop.PILOTS[key]=BINDINGS
    try:
        env=I1StraightRightEnv(key,int(seed),camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        env.reset();assist=make_assist(env)
        # Existing rendered debug cylinder is hidden only in display.
        return env,assist
    finally:
        if existing is None:tabletop.PILOTS.pop(key,None)
        else:tabletop.PILOTS[key]=existing
