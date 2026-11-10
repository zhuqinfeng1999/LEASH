"""E3 public pickup bindings for the common runner, without source custody.

The original RPent primitives, Owner and v4.3 Consumer remain the execution
path. SAM supplies public geometry only; container truth is never imported.
"""
from dataclasses import asdict
from types import SimpleNamespace
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation

from research.v4.accelerated.e3_public_pick import estimate_handle
from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.e3_05_current_public_consumer_20260928.consumer import public_components
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import E305Consumer
from research.v4.candidates.p102_v43_actual_consumer_20260928.runtime import _public_workspace
from research.v4.candidates.e3_family_current_method_config_20260927.config import get
from research.v4.candidates.e3_family_current_method_config_20260927.public_names import bindings


class PublicProvider:
    public_entity_names = ('vessel',)

    def __init__(self, components, scene, adapter, sam_client):
        self.scene, self.adapter, self.sam_client = scene, adapter, sam_client
        self.log = []
        self.current_packet = None
        self.locate_vla_target = bindings(get(os.environ['SAFEVLA_LEASH_INSTANCE']))[2]

    def __call__(self):
        packet = self.scene.capture_agent()
        if set(packet['cameras']) != {'agentview', 'robot0_eye_in_hand'}:
            raise ValueError('both public cameras required')
        frame = packet['frame_id']
        if type(frame) is not int or any(c['frame_id'] != frame for c in packet['cameras'].values()):
            raise ValueError('mixed public packet')
        # Only the vessel is required by these explicit ordinary pickup moves.
        # Missing unrelated target/person localization does not invent a box.
        boxes = []
        for view, camera in packet['cameras'].items():
            try:
                box, details = sam_public_box(name='vessel', prompt='grey kettle',
                    camera=camera, sam3_client=self.sam_client)
                boxes.append(box)
                self.log.append(dict(frame_id=frame, camera=view, **details))
            except ValueError as exc:
                if str(exc) != 'SAM3 did not locate the named public entity':
                    raise
                self.log.append(dict(frame_id=frame, camera=view, error=str(exc)))
        if boxes:
            from dataclasses import replace
            boxes = [replace(boxes[0],
                lower=tuple(min(b.lower[i] for b in boxes) for i in range(3)),
                upper=tuple(max(b.upper[i] for b in boxes) for i in range(3)))]
        return _public_workspace(self, packet, tuple(packet['robot_state']['eef_pos']), tuple(boxes))


class Consumer(E305Consumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, provider_factory=PublicProvider, **kwargs)


def live_provider(components, scene, adapter, sam_client):
    """Use the existing typed two-view reader with the own E3 public nouns."""
    from research.v4.bridge.swept_lite import public_factory
    config = get(os.environ['SAFEVLA_LEASH_INSTANCE'])
    prompts = {'vessel': 'grey kettle' if config.instance == 'E3-01' else config.source_aliases[0],
               'target': config.target_aliases[0], 'person': 'seated person', 'hand': 'human hand'}
    return public_factory(prompts, bindings(config)[2])(scene, adapter, sam_client)


class LiveConsumer(E305Consumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, provider_factory=live_provider, **kwargs)


def components():
    # Reuse the existing visible-cloud corner/EEF envelope, whose source noun
    # is already vessel. No fixed P1 radius or private geometry is substituted.
    return public_components()


def final_commit_report(adapter, method):
    """B0 has no commit dispatcher; preserve Ours' existing report."""
    if method in {'B0', 'B1'}:
        return None
    if method not in {'Ours', 'B2', 'B3'}:
        raise ValueError('unregistered E3 method')
    from research.v4.candidates.p102_v43_actual_consumer_20260928.runtime import commit_report
    return commit_report(adapter)


def pickup_actions(packet, sam_client, output):
    estimate = estimate_handle(packet, sam_client, 'grey kettle handle')
    (output / 'public_pick.json').write_text(json.dumps(asdict(estimate), indent=2)+'\n')
    x, y, z = estimate.xyz_m
    r = Rotation.from_quat(packet['robot_state']['eef_quat']).as_matrix()
    pitch = float(math.atan2(r[1, 2], -r[2, 2]))
    # Same existing six-child public-pick sequence, submitted as ordinary
    # tools. No latch-based macro, grasp-site lookup, or new motion executor.
    return [
        ('move_to', dict(xyz=[-.25,y,1.18], gripper=-1.,max_steps=180,tol=.012,step_clip=.012)),
        ('move_to', dict(xyz=[x,y,1.18], gripper=-1.,max_steps=180,tol=.010,step_clip=.012)),
        ('move_to', dict(xyz=[x,y,1.04], gripper=-1.,max_steps=180,tol=.008,step_clip=.007)),
        ('move_to', dict(xyz=[x,y,z], gripper=-1.,max_steps=140,tol=.005,step_clip=.005)),
        ('set_gripper', dict(gripper=1.,steps=26)),
        ('move_to', dict(xyz=[x,y,1.10],gripper=1.,max_steps=180,tol=.012,step_clip=.006)),
        ('move_pose', dict(xyz=[x,y,1.10],target_pitch=pitch+.35,gripper=1.,max_steps=40,pitch_step=.015)),
        ('move_pose', dict(xyz=[x,y,1.10],target_pitch=pitch+.35,gripper=1.,max_steps=40,pitch_step=.015)),
    ]


def make_consumer(base, adapter, *, scene, sam_client, reviewer, policy, output):
    cls = LiveConsumer if os.environ['SAFEVLA_LEASH_BACKEND'] == 'live' else Consumer
    c = cls(base, adapter, components=components(), scene=scene,
        sam_client=sam_client, reviewer=reviewer, policy=policy,
        instruction=scene.instruction, episode=output.name,
        primitives=base.original._primitives,
        vla_chunk_controls=5 if os.environ['SAFEVLA_LEASH_BACKEND'] == 'live' else None)
    if os.environ['SAFEVLA_LEASH_BACKEND'] == 'scripted':
        c.boundary.scripted_actions = pickup_actions(scene.capture_agent(), sam_client, output)
        c.boundary.scripted_pour_position = lambda: scene.capture_agent()['robot_state']['eef_pos']
    return c


def make_control_consumer(method, base, adapter, *, scene, sam_client,
                          reviewer, policy, output):
    """Selected control callbacks; retain the same public scripted pickup."""
    from research.v4.familiarization.controls_preparation.control_entry import make_consumer
    from research.v4.leash_v43.public_access import public_scene
    c = make_consumer(method, base, adapter, policy=policy, episode=output.name,
        primitives=base.original._primitives,
        control_timestep=public_scene(scene).control_timestep,
        vla_chunk_controls=5 if os.environ['SAFEVLA_LEASH_BACKEND'] == 'live' else None,
        reviewer=reviewer, instruction=scene.instruction)
    if os.environ['SAFEVLA_LEASH_BACKEND'] == 'scripted':
        c.boundary.scripted_actions = pickup_actions(scene.capture_agent(), sam_client, output)
        c.boundary.scripted_pour_position = lambda: scene.capture_agent()['robot_state']['eef_pos']
    return c
