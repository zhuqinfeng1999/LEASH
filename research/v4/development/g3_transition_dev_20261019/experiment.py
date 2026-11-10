"""Three fixed natural development checks; finite, sequential, never a daemon."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from .scope import VERSION

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / 'research/v4/results/g3_transition_dev_20261019'
PLAN = HERE / 'plan.json'
RUNS = ROOT / 'research/v4/runs/g3_transition_dev_20261019'
CODEX = 'release-user/.codex/packages/standalone/releases/0.160.0-x86_64-unknown-linux-musl/bin/codex'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.g3transition.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def check_formal_paused():
    for campaign in ('formal_confidence_20261008', 'formal_visual_backends_20261019'):
        state = read(ROOT / 'research/v4/results' / campaign / 'plan.json').get('authorization', {}).get('state', '')
        if not state.startswith('PAUSED'):
            raise RuntimeError(f'{campaign} is not explicitly paused: {state}')


def register():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    from research.v4.development.confidence_gate_dev_20261006.native import SOURCES
    package = Path(__file__).parent
    required = [package / name for name in ('__init__.py', 'scope.py', 'native.py')]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Isolated runtime not complete; do not freeze yet: ' + str(missing))
    files = sorted(set(SOURCES) | {str(p.relative_to(ROOT)) for p in package.glob('*.py')}
                   | {str(p.relative_to(ROOT)) for p in package.glob('prompt*.txt')})
    episodes = []
    for index, instance in enumerate(('P2-01', 'P3-05', 'S3-01'), 1):
        rid = f'G3T{index:02d}_{instance.replace("-", "")}_OursLuna_seed0_20261019'
        episodes.append(dict(run_id=rid, index=index, instance=instance, method='Ours-conf', seed=0,
            phase='development', formal=False, condition='isolated_g3_transition_scope',
            configuration_profile='current_formal_shared_configuration',
            user_review_protocol='user-review-dev6-local-recovery-20261008',
            placement_task_version='p2-uncued-right-hand-table-dev9-20261014',
            trial_protocol_version=VERSION, planner_model='gpt-6-astra', reasoning='high',
            transition_version=VERSION,
            reviewer_backend='luna', reviewer_model='gpt-6-luna', reviewer_reasoning='high', thresholds=[.5, .5, .5],
            wall_limit_s=5400, physical_gpu=1, attempt_number=1,
            output=str(RUNS / rid), status='registered'))
    plan = dict(version=VERSION, registered_utc=utc(), formal=False, source_custody=False,
        authority='2026-10-19 direct user: improve G3 and test P2-01/P3-05/S3-01; inspect G2; preserve prior results',
        authorization=dict(state='REGISTERED_DEVELOPMENT_ONLY', maximum_natural_starts=3,
                           enabled_backends=['luna'], paid_backends_forbidden=['jev','openai_decisions']),
        no_automatic_retry=True, single_robot=True, formal_allowed=False,
        historical_comparison='Previous JEV trajectories are call-density references only; reviewer latency is not backend-matched. No causal paired speedup claim.',
        limits=dict(primitives_per_round=60, primitives_per_episode=180, planner_turns_per_round=80,
                    wall_s=5400, output_bytes=1073741824, vla_chunk_controls=5),
        policy_sha256='58b612f4c16238d52aab94bba6d90e666cf29d0606b2c92d55db6c8c39157815',
        examples_sha256='7994c71d22d8beacf676157f14158d13185110e6e20f5817a5dcde8c6e588462',
        source_freeze={name: sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        episodes=episodes)
    HERE.mkdir(parents=True, exist_ok=True)
    write(PLAN, plan)
    publish(plan)
    return plan


def publish(plan):
    """Publish only this named question; leave formal/efficiency/G2 data intact."""
    status_path = ROOT / 'research/v4/results/status.json'
    status = read(status_path)
    q = status.setdefault('research_questions', {})
    active = [r for r in plan['episodes'] if r['status'] == 'starting' and r.get('pid')]
    attempts_ended = sum(r['status'] not in ('registered', 'starting') for r in plan['episodes'])
    ended = len({r['instance'] for r in plan['episodes'] if r.get('result_valid') is True})
    old = q.get('g3_transition_development', {})
    data = dict(plan=str(PLAN.relative_to(ROOT)), planned=3, ended=ended,
        attempts_ended=attempts_ended, attempts_registered=len(plan['episodes']),
        active=[dict(run_id=r['run_id'], pid=r['pid'], instance=r['instance']) for r in active],
        authorization_state=plan['authorization']['state'], formal_paused=True,
        summary_path=str((HERE / 'summary.json').relative_to(ROOT)), updated_utc=utc())
    if 'summary' in old:
        data['summary'] = old['summary']
    q['g3_transition_development'] = data
    item = dict(id='g3-transition-development', title='G3 提交过程优化：固定三任务自然验证',
        state=f'{ended}/3 已结束；' + (', '.join(r['run_id'] for r in active) if active else '当前无活动机器人'),
        facts='P2-01、P3-05、S3-01各seed0一次；Astra/high、Luna/high、τ=.5；禁用付费JEV/Decisions测试，正式暂停。',
        analysis='比较当前有界提交过程与历史逐控制审查密度；原始失败与历史视频保留。不把提前退出当提速。',
        next_action='按固定清单完成完整回合，核对提交次数、许可复用、安全与真实完成阶段。',
        links=[dict(label='固定清单', path=str(PLAN.relative_to(ROOT))),
               dict(label='对照报告', path=str((HERE / 'report.md').relative_to(ROOT))),
               dict(label='G2 输入/派发审计', path=str((HERE / 'g2_audit_receipt.md').relative_to(ROOT)))],
        prompt='依据实际完整轨迹比较G3事件/许可粒度和任务表现；保留真实拒绝，不能用少调用替代安全验证。')
    previous = next((i for i in q.get('items', []) if i.get('id') == item['id']), {})
    item['links'] += [link for link in previous.get('links', [])
                      if link.get('label', '').startswith('新回放 ')]
    q['items'] = [i for i in q.get('items', []) if i.get('id') != item['id']] + [item]
    q['updated_utc'] = utc()
    status['updated_utc'] = utc()
    write(status_path, status)


def run():
    plan = read(PLAN)
    check_formal_paused()
    if any(row['status'] == 'starting' for row in plan['episodes']):
        raise RuntimeError('A started attempt requires inspection; never replay a starting row')
    progress = ROOT / 'research/v4/development/transport_workspace_dev_20261005/progress.json'
    if read(progress).get('stop_requested'):
        raise RuntimeError('Root stop flag is still set; this driver never clears it automatically')
    plan['authorization']['state'] = 'RUNNING_DEVELOPMENT_ONLY'
    write(PLAN, plan)
    for row in plan['episodes']:
        if row['status'] != 'registered':
            continue
        check_formal_paused()
        if read(progress).get('stop_requested'):
            plan['authorization']['state'] = 'PAUSED_BY_ROOT_STOP'
            write(PLAN, plan)
            publish(plan)
            return
        drift = [name for name, digest in plan['source_freeze'].items()
                 if sha256((ROOT / name).read_bytes()).hexdigest() != digest]
        if drift:
            raise RuntimeError('Frozen isolated runtime drift: ' + str(drift))
        if Path(row['output']).exists():
            raise FileExistsError('Output already exists; do not overwrite or replay: ' + row['output'])
        row.update(status='starting', started_utc=utc(), pid=None)
        write(PLAN, plan)
        publish(plan)
        command = [str(ROOT / 'envs/libero/bin/python'), '-u', '-m',
            'research.v4.development.g3_transition_dev_20261019.native',
            '--method', 'Ours-conf', '--instance', row['instance'], '--seed', '0', '--gpu', '1',
            '--planner-model', 'gpt-6-astra', '--reviewer-backend', 'luna',
            '--plan', str(PLAN), '--output', row['output'],
            '--tau-task', '.5', '--tau-action', '.5', '--tau-commit', '.5']
        # No paid backend is launched by this fixed development entry.
        env = dict(os.environ, CODEX_BIN=CODEX,
            PYTHONPATH=str(ROOT) + ':' + str(ROOT / 'upstream/RPent-reproduce-libero'),
            PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
        started = time.monotonic()
        try:
            with (HERE / (row['run_id'] + '.log')).open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                row['pid'] = child.pid
                write(PLAN, plan)
                publish(plan)
                print(row['run_id'], 'PID', child.pid, flush=True)
                code = child.wait()
            row.update(process_wall_s=time.monotonic() - started, exit_code=code,
                       ended_utc=utc(), pid=None)
            output = Path(row['output'])
            ep = read(output / 'episode.json') if (output / 'episode.json').exists() else None
            manifest = read(output / 'run_manifest.json') if (output / 'run_manifest.json').exists() else None
            if ep is not None and manifest is not None and manifest.get('ended_utc'):
                valid = code == 0 and manifest.get('valid') is True
                row.update(status='completed' if valid else 'failed', result_valid=valid,
                    label=ep.get('label'), raw_physical_label=(ep.get('physical') or {}).get('label'),
                    terminal=ep.get('terminal'), episode_wall_s=ep.get('wall_s'),
                    error=ep.get('error') or ep.get('system_failure') or manifest.get('error'))
            else:
                row.update(status='interrupted_or_incomplete', result_valid=False,
                           error='Native exited without a complete episode and ended manifest; preserve unknown tail')
        except BaseException as exc:
            # If the parent is interrupted while native is alive, preserve its
            # starting PID for inspection; never pretend the robot finished.
            if row.get('pid'):
                row['parent_interrupted'] = dict(kind=type(exc).__name__, utc=utc())
            else:
                row.update(status='failed_before_launch', result_valid=False,
                           ended_utc=utc(), error=type(exc).__name__ + ': ' + str(exc))
            write(PLAN, plan)
            publish(plan)
            raise
        write(PLAN, plan)
        publish(plan)
        from .report import summarize
        summarize()
        print(row['run_id'], row['status'], row.get('label'), round(row['process_wall_s'], 2), flush=True)
        if row['status'] != 'completed':
            plan['authorization']['state'] = 'HALTED_FOR_TECHNICAL_INSPECTION'
            write(PLAN, plan)
            publish(plan)
            return
    plan['authorization']['state'] = 'COMPLETE_DEVELOPMENT_ONLY'
    write(PLAN, plan)
    publish(plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('register', 'run', 'report'))
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.action == 'run':
        run()
    else:
        from .report import summarize
        summarize()
