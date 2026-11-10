"""Private expert waypoint seam only; original actual Scene/driver/cases."""
import ast
import hashlib
import json
from pathlib import Path
from types import FunctionType

from research.v4.candidates.r305_top_minimum_clearance_20260928 import prepare as parent
from research.v4.candidates.r305_panel_normal_tangent_saved_copy_v2_20260928.check import front_normal_then_tangent

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
VERSION = 'R305-standard-top-derived-clearance-panel-normal-tangent-OSC-v3'
FORWARD_EXCURSION_M = .036734275308400836
BEFORE = "    pose(outside,A,'radial_lowcarry_mouth_alignment',grip=1.,max_steps=220)\n"
AFTER = (
    "    panel_plan = front_normal_then_tangent(m, d, solid, d.site_xpos[center].copy(), outside,\n"
    "        position_stop_m=.008, rotation_stop_rad=.03, measured_forward_excursion_m=FORWARD_EXCURSION_M)\n"
    "    details['actual_front_panel_clearance'] = dict(panel_plan, actual_start_sequence=int(scene.sequence), private_offline_only=True)\n"
    "    pose(np.asarray(panel_plan['retreat_EEF_world_m']), A, 'actual_front_panel_normal_retreat', grip=1., max_steps=220)\n"
    "    outside = np.asarray(panel_plan['tangent_EEF_world_m'])\n"
    "    pose(outside, A, 'actual_front_panel_tangent_alignment', grip=1., max_steps=220)\n")


def expert():
    old = parent.actual_expert.SELECTED_NAIVE_SOURCE
    if old.count(BEFORE) != 1:
        raise RuntimeError('Original own lateral call seam differs')
    source = old.replace(BEFORE, AFTER, 1)
    namespace = dict(parent.naive().__globals__, front_normal_then_tangent=front_normal_then_tangent,
        FORWARD_EXCURSION_M=FORWARD_EXCURSION_M)
    exec(compile(source, str(H / 'entry.py') + '::original-own-OSC-waypoints', 'exec'), namespace)
    fn = namespace['run_naive']
    if fn.__globals__['generic_driver_source'] is not parent.naive().__globals__['generic_driver_source']:
        raise RuntimeError('Original driver changed')
    return fn, source


def selected():
    original, source = parent.selected()
    if source.count(parent.VERSION) != 1:
        raise RuntimeError('Existing case version seam differs')
    source = source.replace(parent.VERSION, VERSION)
    fn, _ = expert()
    namespace = dict(original.__globals__, run_naive=fn)
    exec(compile(source, str(H / 'entry.py') + '::same-original-case', 'exec'), namespace)
    for name in ('R3TabletopFacade', 'Evaluator', 'MAX_STEPS', 'MAX_WALL_S'):
        if namespace['run_case'].__globals__[name] is not original.__globals__[name]:
            raise RuntimeError('Original Scene/evaluator/budget changed: ' + name)
    return namespace['run_case'], source


def admission():
    pins = json.loads((H / 'source_freeze.json').read_text())
    for name, digest in pins.items():
        p = Path(name) if name.startswith('/') else ROOT / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Exact own source drift ' + name)
    parent.admission()
    proof = json.loads((H.parent / 'r305_panel_normal_tangent_saved_copy_v3_20260928/report.json').read_text())
    if not proof['passed'] or proof['failure'] or not proof['actual_copied_center_critical']['critical']:
        raise RuntimeError('Own continuous copy prerequisite unavailable')
    if any(c['hard_stop'] for row in proof['rows'] for c in row['contacts']):
        raise RuntimeError('Own copied hard contact')
    return pins


def preflight():
    pins = admission()
    fn, source = selected()
    naive, expert_source = expert()
    if source != (H / 'selected_case_source.txt').read_text() or expert_source != (H / 'selected_expert_source.txt').read_text():
        raise RuntimeError('Selected frozen source differs')
    if fn.__globals__['MAX_STEPS'] != 1200 or fn.__globals__['MAX_WALL_S'] != 90.:
        raise RuntimeError('Original budget changed')
    writes = []
    for node in ast.walk(ast.parse(expert_source)):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if any(k in ast.unparse(target) for k in ('qpos', 'eq_data', 'eq_active', 'ctrl[')):
                    writes.append(ast.unparse(target))
    if writes:
        raise RuntimeError('Unexpected actual expert live state writes ' + str(writes))
    return dict(passed=True, source_count=len(pins), Scene_main_controls_GPU_models=0,
        actual_driver_position_stop_m=.008, copied_position_stop_m=.012,
        original_caps=[1200, 90], common_reset_controls=20, state_writes=writes,
        actual_scene_identity_retained=True, true_route_unrun=True)


def main():
    dispatch = parent.actual_entry.main
    fn = FunctionType(dispatch.__code__, dict(dispatch.__globals__, H=H, VERSION=VERSION,
        admission=admission, preflight=preflight, selected=selected,
        SELECTED_NAIVE_SOURCE=expert()[1]), dispatch.__name__, dispatch.__defaults__, dispatch.__closure__)
    return fn()


if __name__ == '__main__':
    main()
