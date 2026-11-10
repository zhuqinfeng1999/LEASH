"""One bounded native P3 coaster-contact diagnostic, not task acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from .evaluator import _world_aabb
from .scene import COASTER, CUP, P3Scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = None
    start = time.monotonic()
    report = {'task': 'P3', 'owner': args.output.name, 'cup_asset': 'cup65',
              'seed': args.seed, 'pid': os.getpid(), 'control_limit': 20,
              'task_acceptance': False, 'status': 'failed', 'rows': []}
    try:
        scene = P3Scene(seed=args.seed, size=256, cup_asset='cup65')
        (args.output / 'scene.xml').write_text(scene.source_scene_xml)
        report['source_hashes'] = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ('scene.py', 'evaluator.py', 'coaster.xml', 'fixture_diagnostic.py', 'scene.bddl')}
        pmon = subprocess.run(['nvidia-smi', 'pmon', '-c', '2', '-s', 'um'],
                              capture_output=True, text=True, timeout=8, check=True)
        report['host_pmon'] = pmon.stdout
        report['physical_gpu1_pid_observed'] = any(
            len(parts) > 1 and parts[0] == '1' and parts[1] == str(os.getpid())
            for parts in (line.split() for line in pmon.stdout.splitlines()))
        if not report['physical_gpu1_pid_observed']:
            raise RuntimeError('P3 fixture PID not observed on host physical GPU1; no controls sent')
        sim = scene.env.sim
        coaster = scene.env.env.objects_dict[COASTER]
        cup = scene.env.env.objects_dict[CUP]
        cg = sim.model.geom_name2id(coaster.contact_geoms[0])
        cup_ids = [sim.model.geom_name2id(name) for name in cup.contact_geoms]
        report['compiled_coaster_solref'] = np.asarray(sim.model.geom_solref[cg]).tolist()
        report['compiled_coaster_solimp'] = np.asarray(sim.model.geom_solimp[cg]).tolist()
        report['compiled_coaster_priority'] = int(sim.model.geom_priority[cg])
        report['coaster_geom_name'] = coaster.contact_geoms[0]
        report['table_top_site_z'] = float(sim.data.site_xpos[sim.model.site_name2id('table_top')][2])

        def sample(step):
            sim.forward()
            coaster_box = _world_aabb(sim, cg)
            cup_box = [_world_aabb(sim, gid) for gid in cup_ids]
            contacts = []
            for c in sim.data.contact[:sim.data.ncon]:
                a = sim.model.geom_id2name(c.geom1) or ''
                b = sim.model.geom_id2name(c.geom2) or ''
                if cg in (int(c.geom1), int(c.geom2)):
                    contacts.append({'a': a, 'b': b, 'distance_m': float(c.dist)})
            report['rows'].append({
                'control_step': step, 'sim_time_s': float(sim.data.time),
                'coaster_qpos_m': np.asarray(sim.data.get_joint_qpos(coaster.joints[0]))[:3].tolist(),
                'coaster_collision_bottom_z_m': float(coaster_box[0][2]),
                'coaster_collision_top_z_m': float(coaster_box[1][2]),
                'cup_qpos_m': np.asarray(sim.data.get_joint_qpos(cup.joints[0]))[:3].tolist(),
                'cup_collision_bottom_z_m': float(min(x[0][2] for x in cup_box)),
                'coaster_contacts': contacts,
            })

        sample(0)
        packet = scene.observation()
        for name, view in packet['cameras'].items():
            Image.fromarray(view['rgb']).save(args.output / f'initial-{name}.png')
        Image.fromarray(scene.spectator_rgb()).save(args.output / 'initial-spectator.png')
        for step in range(1, 21):
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.
            scene.step(action)
            sample(step)
            if time.monotonic() - start > 100:
                raise TimeoutError('P3 fixture diagnostic wall bound')
        packet = scene.observation()
        for name, view in packet['cameras'].items():
            Image.fromarray(view['rgb']).save(args.output / f'final-{name}.png')
        Image.fromarray(scene.spectator_rgb()).save(args.output / 'final-spectator.png')
        report['status'] = 'completed_20_controls'
    except BaseException as exc:
        import traceback
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
    finally:
        if scene is not None:
            scene.close()
        report['elapsed_s'] = time.monotonic() - start
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'output': str(args.output), 'status': report['status'],
                          'physical_gpu1_pid_observed': report.get('physical_gpu1_pid_observed'),
                          'error': report.get('error')}), flush=True)
    if report['status'] != 'completed_20_controls':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
