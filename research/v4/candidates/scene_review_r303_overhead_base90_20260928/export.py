"""Thin R303 qualified-version native HD and two-public-view evidence binding."""
import hashlib
import json
import types
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from PIL import Image

from research.v4.candidates.scene_review_r303_horizontal_20260928 import export as writer
from research.v4.candidates.r303_seed2_overhead_actual_20260928 import run_once as version
from research.v4.candidates.r303_overhead_qualification10_20260928 import qualify
from research.v4.candidates.r303_base_forward90_copy_20260928.scene import BaseForward90OwnThermosScene

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
FINAL = H / 'R3-03.overhead-base90.preview.mp4'
QUALIFICATION = ROOT / 'research/v4/candidates/r303_overhead_qualification10_20260928/root_qualification10/summary.json'
TASK = writer.TASK


def state(scene):
    m, d = scene.backend.sim.model._model, scene.backend.sim.data._data
    s = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(m, s))
    mujoco.mj_getState(m, d, result, s)
    return result


def public_snapshot(scene, stage):
    """Offline artifacts from the actual existing public getter; no controls."""
    if not hasattr(scene, '_review_chapter') or stage in scene._review_saved:
        return
    before = state(scene)
    m = scene.backend.sim.model._model
    ids = [scene.backend.sim.model.camera_name2id(n) for n in scene.public_views]
    cameras = {k: np.asarray(getattr(m, k)[ids]).copy() for k in ('cam_pos', 'cam_quat', 'cam_fovy')}
    packet = scene.capture_agent()
    if not np.array_equal(before, state(scene)):
        raise RuntimeError('Public evidence capture changed integration state')
    if not all(np.array_equal(v, np.asarray(getattr(m, k)[ids])) for k, v in cameras.items()):
        raise RuntimeError('Public evidence capture changed camera model')
    if set(packet['cameras']) != {'agentview', 'robot0_eye_in_hand'}:
        raise RuntimeError('Public evidence camera protocol changed')
    folder = H / 'capture' / 'public' / f'chapter{scene._review_chapter}' / stage
    folder.mkdir(parents=True, exist_ok=False)
    views = {}
    for name, camera in packet['cameras'].items():
        rgb, depth = np.asarray(camera['rgb']), np.asarray(camera['depth_m'])
        if rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
            raise RuntimeError('Actual public RGB-D dimensions differ')
        Image.fromarray(rgb).save(folder / (name + '.png'))
        np.savez_compressed(folder / (name + '.npz'), depth_m=depth,
                            intrinsics=np.asarray(camera['intrinsics']),
                            extrinsics=np.asarray(camera['extrinsics']))
        views[name] = {'frame_id': camera['frame_id'], 'rgb_shape': list(rgb.shape),
                       'depth_shape': list(depth.shape),
                       'K': np.asarray(camera['intrinsics']).tolist(),
                       'T': np.asarray(camera['extrinsics']).tolist()}
    robot = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in packet['robot_state'].items()}
    metadata = {'stage': stage, 'sequence': scene.sequence,
                'simulation_time_s': float(scene.backend.sim.data.time),
                'camera_protocol': 'qualified base-forward90 two native640 RGB-D views',
                'cameras': views, 'robot_state': robot,
                'initial_base_table_check': scene._review_base_table,
                'integration_state_equal': True, 'public_camera_model_equal': True,
                'offline_only': True}
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    scene._review_saved.add(stage)


class EvidenceScene(BaseForward90OwnThermosScene):
    def __init__(self, seed=0, *, camera_observations=False, render_gpu_device_id=-1):
        super().__init__(seed, camera_observations=camera_observations,
                         render_gpu_device_id=render_gpu_device_id,
                         root_base_forward_approved=True)
        self._review_saved = set()
        m, d = self.backend.sim.model._model, self.backend.sim.data._data
        self._review_initial_height = float(d.xpos[m.body('steel_thermos_1_main').id, 2])
        table, link0 = m.geom('table_collision').id, m.geom('robot0_link0_collision').id
        gap = float(mujoco.mj_geomDistance(m, d, table, link0, 10.0, None))
        self._review_base_table = {'base_position_world': m.body_pos[m.body('robot0_base').id].tolist(),
                                   'pair': ['robot0_link0_collision', 'table_collision'],
                                   'actual_distance_m': gap, 'required_clearance_m': .005}
        if gap < .005:
            super().close()
            raise RuntimeError('New base hard link0/table clearance is below original 5 mm check')

    def step(self, action):
        if getattr(self, 'offline_stage', None) == 'actual_common_assist_closure':
            public_snapshot(self, 'before_actual_closure')
        result = super().step(action)
        if self.grasp_assist.held:
            public_snapshot(self, 'actual_attached')
            m, d = self.backend.sim.model._model, self.backend.sim.data._data
            if float(d.xpos[m.body('steel_thermos_1_main').id, 2]) - self._review_initial_height >= .020:
                public_snapshot(self, 'actual_lift20mm')
        return result

    def close(self):
        try:
            public_snapshot(self, 'first_critical_or_final')
        finally:
            super().close()


class ScriptAdapter:
    INSTRUCTION = TASK
    P2Facade = EvidenceScene
    MAX_WALL_S = 90.0

    @staticmethod
    def probe(mode, seed):
        case = {'safe': 'correct', 'naive': 'naive'}[mode]
        _, original, _, naive, _ = version.admission()
        with patch.dict(original.__globals__, {'R3TabletopFacade': ScriptAdapter.P2Facade,
                                              'run_naive': naive}):
            result = original(seed, case)
        result.update(accepted=result['passed'], steps=result['control_steps'], result=result['evaluator'])
        trace = result.get('physical_trace', [])
        bind = result.get('true_bind')
        prebind = [row for row in trace if bind is None or row['sequence'] < bind['sequence']]
        early = [contact for row in prebind for contact in row.get('contacts', [])
                 if contact['dist_m'] <= 0
                 and any(n.startswith('gripper0_') for n in contact['geoms'])
                 and any(n.startswith('steel_thermos_1_') for n in contact['geoms'])]
        hard = [contact for row in trace for contact in row.get('contacts', [])
                if contact['exact_pad_classification']['non_grasp_hard_stop']]
        result['native_contact_observation'] = {'prebind_gripper_thermos_contacts': len(early),
                                                'per_control_nonpad_hard_overlap_count': len(hard),
                                                'full_actual_trace_preserved': True,
                                                'scorer_unchanged': True}
        if result['control_steps'] > 1200:
            raise RuntimeError('Original R303 physical control cap exceeded')
        return result


class Native1080Recorder(writer.Native1080Recorder):
    def begin_segment(self, scene, **kwargs):
        result = super().begin_segment(scene, **kwargs)
        scene._review_chapter = len(self.segment)
        public_snapshot(scene, 'initial')
        return result


def hashes():
    current, _, case_source, _, naive_source = version.admission()
    summary = json.loads(QUALIFICATION.read_text())
    if not summary['complete'] or (summary['k'], summary['n']) != (30, 30):
        raise RuntimeError('Same-version physical 30/30 qualification missing')
    if (summary['source_freeze_sha256'] != qualify.sha(qualify.H / 'source_freeze.json')
            or summary['generated_case_sha256'] != hashlib.sha256(case_source.encode()).hexdigest()
            or summary['overhead_naive_sha256'] != hashlib.sha256(naive_source.encode()).hexdigest()):
        raise RuntimeError('Qualified physical/expert source version differs')
    pins = dict(current)
    for name, digest in writer.hashes().items():
        if name in pins and pins[name] != digest:
            raise RuntimeError('Original writer/qualified-version source conflict: ' + name)
        pins[name] = digest
    files = [*H.glob('*.py'), H / 'protocol.json', H / 'cpu_contract.json', QUALIFICATION,
             qualify.H / 'source_freeze.json',
             ROOT / 'research/v4/results/r303_overhead_qualification10_cpu_root_review_20260928.json']
    for row in summary['rows']:
        path = Path(row['result_path'])
        if not row['passed'] or qualify.sha(path) != row['result_sha256']:
            raise RuntimeError('Same-version raw qualification changed')
        files.append(path)
    for path in files:
        key = str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)
        pins[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    pins = dict(sorted(pins.items()))
    if (H / 'source_freeze.json').exists() and pins != json.loads((H / 'source_freeze.json').read_text()):
        raise RuntimeError('New R303 native review source drift')
    return pins


def sidecars():
    for path in (H / 'capture/report.json', FINAL.with_suffix('.json')):
        if not path.exists(): continue
        packet = json.loads(path.read_text())
        packet.update(instance_id='R3-03', instruction=TASK,
                      source_code_version='qualified-base90-overhead-expert+original-R303-native-writer',
                      qualification_path=str(QUALIFICATION), qualification_k_n=[30, 30],
                      physical_scene_version=BaseForward90OwnThermosScene.layout_version,
                      per_case_script_caps={'controls': 1200, 'wall_s': 90},
                      public_evidence_folder=str(H / 'capture/public'),
                      no_door_close_heating_or_full_containment_claim=True,
                      method_episode_or_count=False, canonical_replacement=False,
                      prior_layout_outputs_preserved=True,
                      root_encoded_and_public_view_review='pending' if not packet.get('error') else 'failed')
        path.write_text(json.dumps(packet, indent=2) + '\n')


def main():
    # Exact original driver/encoder/process/device/render guards, new bindings
    # only. No copied executor, default Scene mutation, or new control loop.
    namespace = dict(writer.main.__globals__, H=H, FINAL=FINAL, hashes=hashes,
                     ScriptAdapter=ScriptAdapter, Native1080Recorder=Native1080Recorder,
                     sidecars=sidecars)
    types.FunctionType(writer.main.__code__, namespace, writer.main.__name__)()


if __name__ == '__main__':
    main()
