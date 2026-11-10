"""Existing fixed V2 actions through the new confidence runner for calibration.

Only planner external IO is scripted. Current RGB, Owner, command dispatch,
registry and Luna score requests remain real. Zero thresholds collect the
first valid score; malformed scores still withhold. No old geometric Consumer
or diagnostic override is installed. Script coordinates are never memory.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from .native import ROOT


def physical_unreached(raw):
    """Engineering endpoint check from the actual common-control public return."""
    result = (raw.get('log') or {}).get('result', raw)
    description = (result.get('common_motion') or {}).get('description') or {}
    goal = description.get('goal')
    position = (raw.get('state') or {}).get('robot0_eef_pos')
    tolerance = description.get('tol')
    if (result.get('name') == 'move_to' and isinstance(goal, (list, tuple, np.ndarray)) and len(goal) == 3
            and isinstance(position, (list, tuple, np.ndarray)) and len(position) == 3
            and isinstance(tolerance, (int, float)) and tolerance > 0):
        values = [*goal, *position, tolerance]
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return math.dist(goal, position) > tolerance
    return result.get('reached') is False


def failed_return(raw):
    result = (raw.get('log') or {}).get('result', raw)
    return bool(raw.get('system_failure') or raw.get('error') or
                result.get('error') or result.get('success') is False or physical_unreached(raw))


def may_continue_unreached(raw, previous):
    result = (raw.get('log') or {}).get('result', raw)
    return bool(previous and previous.get('continue_after_unreached_engineering')
        and physical_unreached(raw)
        and not (raw.get('system_failure') or raw.get('error') or result.get('error')
                 or result.get('system_failure') or result.get('candidate_dispatched') is False
                 or result.get('blocked') or result.get('intercepted')))


def engineering_hazard(row, case):
    """Existing private evaluator fields, used only to stop an engineering suffix."""
    keys = case.get('offline_predicate_keys', ('harm', 'unsafe_commit', 'critical'))
    return any(row.get(key) is True for key in keys)


def actions(case, holder, journal, output):
    previous = None
    stages = case['stages']
    if case.get('e3_public_pickup'):
        from .prepare_e3_public_pickup import (
            initial_saved_public_packet, prepare_public_pickup, save_public_pick)
        stack=holder['stack']
        sequence=stack.adapter.dispatcher.sequence
        packet=initial_saved_public_packet(stack.base.original.state,executed_sequence=sequence)
        estimate,prefix=prepare_public_pickup(packet,holder['sam3_client'],expected_sequence=sequence)
        save_public_pick(output,estimate)
        stages=[*prefix,*stages]
    for stage in stages:
        if journal and failed_return(journal[-1]['result']) and not may_continue_unreached(journal[-1]['result'], previous):
            return
        private_log = output/'evaluator-only.jsonl'
        if private_log.exists():
            # Some original predicates describe transient physical contact.
            # Its absence in the final frame does not erase an earlier event.
            if any(engineering_hazard(json.loads(line), case)
                   for line in private_log.read_text().splitlines()):
                return  # engineering stopping only; never an online gate input
        arguments = deepcopy(stage['arguments'])
        if 'translation_from_actual_public_p0_m' in stage:
            # The original V2 relative targets use the actual public return.
            robot = holder['stack'].adapter.public_robot()
            arguments['xyz'] = [float(p+d) for p,d in zip(robot['eef_pos'],
                                      stage['translation_from_actual_public_p0_m'])]
        if 'xyz_from_current_public_eef' in stage:
            from .prepare_e12_current_scripts import resolve_public_step
            arguments=resolve_public_step(stage,holder['stack'].adapter.public_robot())['arguments']
        previous = stage
        yield stage['tool'], arguments


def main(plan_path, case_id, output, gpu):
    from research.v4.bridge.run_b0 import bind_rpent_checkout, scripted_external_io
    from research.v4.leash_v43.run_manifest import atomic_json
    from research.v4.development.transport_workspace_dev_20261005 import progress
    from . import native, runner
    plan = json.loads(plan_path.read_text())
    case = next(r for r in plan['episodes'] if r['run_id'] == case_id)
    if case['method'] != 'Ours-conf' or case['thresholds'] != [0.,0.,0.]:
        raise ValueError('Fixed scripted calibration uses confidence shadow only')
    if output.name != case_id or output.exists():
        raise ValueError('Registered fresh output required')
    bind_rpent_checkout()
    holder, journal = {}, []
    if 'rounds' in case:
        from .s_round_stage_actions import RoundStageActions
        actions_override = RoundStageActions(case, holder, journal, output)
    else:
        actions_override = actions(case, holder, journal, output)
    original_stack = runner.create_episode_stack
    original_claim = progress.claim
    @contextmanager
    def stack(**kwargs):
        with original_stack(**kwargs) as current:
            holder['stack'] = current
            holder['sam3_client'] = kwargs['sam3_client']
            # The common runner already records primitive_calls.jsonl through
            # this exact boundary. Do not duplicate the same execution window.
            yield current
    def claim(kind, count=1):
        return original_claim('native_engineering_trajectories' if kind == 'natural_episode_starts' else kind, count)
    code = None
    try:
        with patch.object(runner, 'create_episode_stack', stack), \
             patch.object(progress, 'claim', claim), \
             patch.object(native, 'SOURCES', (*native.SOURCES, str(Path(__file__).relative_to(ROOT)), str(plan_path.relative_to(ROOT)),
                 'research/v4/bridge/run_b0.py','research/v4/development/confidence_gate_dev_20261006/prepare_e3_public_pickup.py',
                 'research/v4/development/confidence_gate_dev_20261006/prepare_e12_current_scripts.py',
                 'research/v4/development/confidence_gate_dev_20261006/s_round_stage_actions.py',
                 'research/v4/development/confidence_gate_dev_20261006/s_round_script_io.py')), \
             scripted_external_io(case['instance'], journal, actions_override=actions_override,
                 finish_status=(case.get('required_terminal') or {}).get('status','done'))():
            code = native.main(SimpleNamespace(output=output, plan=plan_path, instance=case['instance'],
                method=case['method'], seed=case['seed'], gpu=gpu,
                tau_task=0.,tau_action=0.,tau_commit=0.))
    finally:
        if output.exists():
            atomic_json(output/'scripted_tools.json', journal)
            atomic_json(output/'scripted_calibration.json', dict(case=case,return_code=code,
                source_phase='scripted_calibration', natural_method_result=False, formal=False,
                planner_GPT_requests=0, actual_safety_scores=True, geometry_override=False,
                invalid_score_withholds=True, script_not_in_shared_memory=True))
            path = output/'episode.json'
            if path.exists():
                episode = json.loads(path.read_text())
                episode.update(condition='scripted_shadow_calibration', natural_method_result=False,
                    source_phase='scripted_calibration')
                episode['model_calls']['planner_GPT'] = dict(calls=0,elapsed_s=0.,tokens=None,fee='unavailable',backend='scripted_IO')
                size,count=native.output_size(output)
                episode.update(output_bytes=size,output_files=count)
                atomic_json(path,episode)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=1)
    args=parser.parse_args();raise SystemExit(main(args.plan.resolve(),args.case,args.output.resolve(),args.gpu))
