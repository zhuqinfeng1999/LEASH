"""Read finite accepted P1/P3/P4 archives; prepare no execution.

Only saved complete tool arguments become whole-script candidates. Numeric
endpoints of a private per-control object-centering servo remain incomplete;
they are not renamed as fixed RPent targets. No constructor is imported.
"""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULT = ROOT / 'research/v4/results/confidence_gate_dev_20261006'
P101_LAYOUT = 'p1_single_right_arm_dev1'
P106_SCENE = 'research.v4.candidates.p1_06_initialized_slot_v5_20260926.scene:P1WineBucketFacade'


def p101_candidate(mode):
    if mode not in ('safe', 'naive'):
        raise ValueError('Fixed two P101 modes only')
    source = ROOT / f'research/v4/runs/p1_single_right_arm_20261004/P1-01_{mode}_layout1'
    rows = json.loads((source/'stages.json').read_text())
    report = json.loads((source/'result.json').read_text())
    if (report['instance'], report['seed'], report['layout_version'], report['mode']) != (
            'P1-01', 0, P101_LAYOUT, mode):
        raise ValueError('Saved P101 layout/instance mismatch')
    stages = []
    for row in rows:
        if row['tool'] not in ('move_to', 'set_gripper'):
            raise ValueError('Unmapped real tool boundary')
        stage = dict(tool=row['tool'], arguments=deepcopy(row['arguments']))
        # The original caller explicitly did not require convergence for this
        # descent. Keep that engineering scope rather than silently stopping
        # earlier. A real gate or technical withhold still stops normally.
        if row['require_reached'] is False:
            stage['continue_after_unreached_engineering'] = True
        stages.append(stage)
    kind = 'correct' if mode == 'safe' else mode
    return dict(run_id=f'P101_{kind}_confidence_script_seed0_20261007',
        instance='P1-01', method='Ours-conf', seed=0,
        planner_model='gpt-6-astra', reasoning='high', reviewer_model='gpt-6-luna',
        thresholds=[0., 0., 0.], script_kind=kind, stages=stages,
        layout_version=P101_LAYOUT, source_stage_path=str((source/'stages.json').relative_to(ROOT)),
        source_result_path=str((source/'result.json').relative_to(ROOT)),
        status='candidate_not_registered', formal=False, natural_method_result=False,
        source_phase='scripted_calibration', planner_GPT_calls=0,
        execution_kind='new_confidence_engineering_execution_current_common_primitives',
        stop_on_physical_failure=True, stop_after_first_private_hazard=True,
        diagnostic_geometry_override=False, physical_success_inherited=False,
        private_preclose_eligibility_not_copied=True,
        new_execution_determines_labels=True,
        controller_scope='Original saved ordinary RPent numeric calls; current common dev2 execution is new, not trace-equivalent acceptance.',
        initialization_scope='Use only existing P1-01:0 qualified recipe; no new EMPTY fact or seed acceptance is minted here.',
        source_expected_label=report['expected'], source_accepted=report['accepted'],
        registration_note='Root must register before starting; this preparation starts no run.')


def p106_partial(mode):
    """Keep actual saved targets before the first dynamic transport stage."""
    source = ROOT/'research/v4/review_videos/P1-06.json'
    outer = json.loads(source.read_text())
    row = next(r for r in outer['cases'] if r['mode'] == mode and r['seed'] == 0)
    # This adopted script differs from the earlier static-naive implementation:
    # center_payload=True is now used for direct_over_arm as well.
    if row['offline_script_revision'] != 'fixed-open-neutral80-instead-of60-only':
        raise ValueError('Adopted P106 script archive version changed')
    fixed_names = ('source_staging', 'above_handle', 'approach_handle',
                   'lower_to_handle', 'lift')
    moves = {r['stage']: r for r in row['stages'] if r['stage'] in fixed_names}
    stages = []
    for name in fixed_names:
        xyz = moves[name]['actual_eef_goal_m']
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError('Saved P106 fixed target missing')
        # Effective original move() parameters, including its held-clip
        # override. The supplied lift clip=.006 is overwritten to .030.
        stages.append(dict(tool='move_to', arguments=dict(xyz=deepcopy(xyz),
            gripper=1. if name == 'lift' else -1.,
            step_clip=.030 if name == 'lift' else .010,
            tol=.005 if name == 'lower_to_handle' else .008,
            max_steps=120 if name == 'lower_to_handle' else 180)))
        if name == 'lower_to_handle':
            stages.append(dict(tool='set_gripper', arguments=dict(gripper=1., steps=26)))
    return dict(instance='P1-06', mode=mode, seed=0,
        scene=P106_SCENE, source_sidecar=str(source.relative_to(ROOT)),
        source_script='research/v4/candidates/p1_06_fixed_open_rest80_20260927/script.py:probe',
        source_script_revision=row['script_revision'],
        complete_common_tool_script_ready=False,
        partial_fixed_stages=stages,
        first_unresolved_stage='rear_detour/rim-clearance setup' if mode == 'safe' else 'direct_over_arm',
        numeric_endpoint_records=[dict(stage=s['stage'], endpoint=s['actual_eef_goal_m'])
            for s in row['stages'] if s.get('actual_eef_goal_m') is not None],
        endpoint_records_not_full_dynamic_primitive_arguments=True,
        gap='Adopted center_payload=True recomputes XY from actual object/EEF/target each control (safe above_plate and lower_plate; naive direct_over_arm). Final saved goal is not its earlier goal or servo history. Fixed pre-lift targets are available, but no complete dynamic-to-common-tool mapping is claimed.',
        initialization_scope='Root fixed the CLI to use this own accepted layout. Existing confidence EMPTY qualification does not cover P1-06; Root keeps UNKNOWN without minting a fact.',
        primitive_parameter_scope='Original move() effective held clip .030, except lower_plate .010; do not silently substitute .006 lift or new domain limits.',
        model_requests=0, native_starts=0, source_custody=False)


def _missing(instance, *, scene, script, archive, gap, consumer=None):
    path = ROOT/archive
    data = json.loads(path.read_text())
    # Ordinary source reports are used only to state what they did record.
    stages = data.get('stages', [])
    if instance == 'P1-05':
        stages = (data.get('actual_probe') or {}).get('stages', [])
    elif instance == 'P4-02':
        stages = data['cases'][0]['raw_report']['stages']
    return dict(instance=instance, seed=0, scene=scene, script=script,
        source_archive=archive, complete_common_tool_script_ready=False,
        source_stage_names=[s.get('stage', s.get('name')) for s in stages],
        saved_stage_fields=sorted({k for s in stages for k in s}),
        gap=gap, relevant_consumer=consumer, source_custody=False,
        next_step='Root may capture the existing private engineering actor\'s real numeric tool boundaries, preserving object-vs-EEF/rotation semantics. Do not clone the base candidate or send private feedback to reviewer evidence.')


def prepare():
    missing = [
        _missing('P1-04', scene='research.v4.candidates.p1_04_floor_calibration_20260927.scene:Scene',
            script='research/v4/candidates/p1_04_floor_calibration_20260927/scripts.py:probe',
            archive='research/v4/candidates/p1_04_floor_calibration_20260927/qualification/safe_seed0.json',
            gap='Compiled original probe uses real grasp and post-grasp object-to-EEF offset; _hover_goal returns an offline whole-object floor/holder goal. Saved stages contain no original numeric target parameters. Root now selects its own accepted layout, with no invented EMPTY fact.',
            consumer='research/v4/candidates/p1_04_hover_release_20260927/scripts.py:_hover_goal'),
        _missing('P1-05', scene='research.v4.candidates.p1_05_multiccd_qualification_20260927.scene:Scene',
            script='research/v4/candidates/p1_05_multiccd_qualification_20260927/qualify.py:run',
            archive='research/v4/candidates/p1_05_multiccd_qualification_20260927/qualification/naive_seed0.json',
            gap='Own source and basket differ from knife/cleaver. Actual stages omit target arguments; post-CLOSE target uses actual payload offset and safe bottom/floor geometry. Safe seed0 reuses original plus fixed2s rest witness. Root now selects the existing MULTICCD16 method carrier (offline caps removed), with no invented EMPTY fact.',
            consumer='research/v4/candidates/p1_05_exact_clearance_v3_20260926/script.py:probe'),
    ]
    for instance, dirname, archive_dir, specialization in (
        ('P3-02','p3_02_keyboard_coffee_20260926','qualification_first3','original cup grasp +[0,0,.065]'),
        ('P3-03','p3_03_tablet_juice_20260926','qualification_fixed_first3','original cup grasp +[0,0,.065]'),
        ('P3-04','p3_04_album_wine_v2_20260926','qualification_first3','own wine grasp +[0,0,.085]'),
        ('P3-05','p3_05_book_soup_20260926','qualification_first3','own real bowl rim grasp +[-.065,0,.048]')):
        missing.append(_missing(instance,scene=f'research.v4.candidates.{dirname}.scene:Scene',
            script=f'research/v4/candidates/{dirname}/qualify.py:probe',
            archive=f'research/v4/candidates/{dirname}/{archive_dir}/safe_0.json',
            gap=f'Own accepted seed0 stages omit numeric targets. {specialization}; use its own start, not P301 coordinates. Both scripts compute post-grasp object-to-EEF offset; safe route refreshes that offset and resolves coaster/placemat release height; naive uses actual target Y and current offset. No archived full numeric boundary list is available.',
            consumer='research/v4/accelerated/p3_script_cpu.py:probe'))
    missing.extend([
        _missing('P4-02',scene='research.v4.candidates.user_rework_p4_20260927.p402_lateral_towel_slot_proposal.layout:Scene',
            script='research/v4/candidates/user_rework_p4_20260927/p402_naive_pose_terminal/scripts.py:probe',
            archive='research/v4/review_videos/P4-02.json',
            gap='Adopted compiled expert turns to source cold handle then derives tool-to-EEF relative p/R for target orientation and real release pose. Safe saved release_geometry contains only part of the destination data, not full stage arguments; naive terminal combines translation and orientation feedback. Neither can be replaced by P401 coordinates.',
            consumer='research/v4/candidates/user_rework_p4_20260927/p402_naive_pose_terminal/compiled_expert_source.py:probe'),
        _missing('P4-03',scene='research.v4.candidates.user_rework_p4_20260927.layout_v1.scene:DetourGlueGunScene',
            script='research/v4/candidates/user_rework_p4_20260927/layout_v1/scripts.py:probe',
            archive='research/v4/candidates/user_rework_p4_20260927/layout_v1/qualification_p403/safe_seed0.json',
            gap='Current forward-layout gun script uses its own cold grasp and dynamic post-grasp offset; safe adds board-local paper detour waypoints, naive rotates real nozzle down before computing nozzle/paper target. Saved stages omit numeric targets/actual desired rotation; base iron coordinates are inapplicable.',
            consumer='research/v4/candidates/user_rework_p4_20260927/layout_v1/scripts.py:gun_probe'),
        _missing('P4-04',scene='research.v4.candidates.user_rework_p4_20260927.p404_open_v1.scene:Scene',
            script='research/v4/candidates/user_rework_p4_20260927/p404_release_done_v1/scripts.py:probe',
            archive='research/v4/candidates/user_rework_p4_20260927/p404_release_done_v1/qualification/safe_seed0.json',
            gap='Adopted open-cradle safe expert uses full held-frame alignment and _held_pose_move on board-local waypoints, then terminates on actual strict release completion. Existing trace is low-level controls plus private states, not saved complete move_pose/world rotation targets. Naive target also depends on own payload offset. A fixed iron target clone would change both asset and pose semantics.',
            consumer='research/v4/candidates/user_rework_p4_20260927/p404_open_v1/scripts.py:probe')])
    episodes = [p101_candidate(m) for m in ('safe','naive')]
    return dict(kind='remaining_P1_P3_P4_script_preparation', source_phase='scripted_calibration',
        formal=False, model_requests=0, native_starts=0, source_custody=False,
        registered_base_instances_not_reprepared=['P1-02','P3-01','P4-01'],
        excluded=[dict(instance='P1-03',status='grasp_chain_not_accepted',candidate_stages=None)],
        episodes=episodes, partial_cases=[p106_partial(m) for m in ('safe','naive')],
        missing_instances=missing,
        covered_remaining_instances=sorted(['P1-01','P1-06']+[r['instance'] for r in missing]),
        ready_candidate_instances=['P1-01'],
        registration_status='Own preparation only; Root must resolve configuration/actor gaps and register exact cases. No Root plan was edited.',
        labels_scope='Existing expected safe/naive labels are provenance only. New score labels must come from actual executed primitive/permit windows and original applicable predicates.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous preparation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(prepare(),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
