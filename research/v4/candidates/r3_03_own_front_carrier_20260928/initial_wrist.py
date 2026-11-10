"""Explicit disabled-by-default designed initial reset variant, never runtime fix."""
import math
from unittest.mock import patch
from research.v4.candidates.r3_03_t1_derived_20260926 import scene as own
from .scene import OwnFrontThermosScene
ORIGINAL_NEUTRAL=tuple(own.NEUTRAL_QPOS)
INITIAL_JOINT7_PLUS_PI=ORIGINAL_NEUTRAL[:-1]+(ORIGINAL_NEUTRAL[-1]+math.pi,)

class InitialWristMirrorFrontThermosScene(OwnFrontThermosScene):
    layout_version='R303-own-front-one-designed-initial-joint7-plus-pi-pending-v1'
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1,
                 root_initial_wrist_approved=False):
        if root_initial_wrist_approved is not True:
            raise ValueError('Initial wrist variant is disabled pending explicit root review')
        # Own original constructor reads this tuple when it sets PARK_QPOS
        # before robot._load_model / ordinary initialization. No live arm
        # qpos, actuator/controller, object/weld pose or tail is rewritten.
        with patch.object(own,'NEUTRAL_QPOS',INITIAL_JOINT7_PLUS_PI):
            super().__init__(seed,camera_observations=camera_observations,
                             render_gpu_device_id=render_gpu_device_id)

def selected_scene(instance_id='R3-03',*,initial_wrist_plus_pi=False,
                   root_initial_wrist_approved=False):
    if instance_id!='R3-03':raise ValueError('Own exact instance only')
    if initial_wrist_plus_pi:
        if root_initial_wrist_approved is not True:raise ValueError('Unapproved initial wrist variant')
        return InitialWristMirrorFrontThermosScene
    return OwnFrontThermosScene
