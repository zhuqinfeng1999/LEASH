"""Finite P4-02..04 ordinary business capture; no scene or model starts.

Original full-SO3, held-object-frame helpers, and inline controller gaps are
retained as unmapped. Only actual move/hold arguments and optional same-return
public EEF pose are recorded. No source/pins/AST/callable/control stream is
saved or reconstructed by this recorder.
"""
from copy import deepcopy
import sys

from .collect_e12_movement_parameters import NumericBoundaryRecorder
from .p2_business_parameters import public_pose_reference
from .s12_business_parameters import BusinessParameterCapture, _count, _number


P4_SCRIPTS = {
    'P4-02': 'research.v4.candidates.user_rework_p4_20260927.p402_naive_pose_terminal.scripts',
    'P4-03': 'research.v4.candidates.user_rework_p4_20260927.layout_v1.scripts',
    'P4-04': 'research.v4.candidates.user_rework_p4_20260927.p404_release_done_v1.scripts',
}
BUSINESS_MODULES = (
    'research.v4.candidates.p4_02_hot_end_edge_expert_v11_20260927.script_cpu',
    'research.v4.candidates.p4_03_gluegun_20260926.scripts',
    'research.v4.candidates.p4_04_soldering_iron_20260926.scripts',
    'research.v4.candidates.p4_04_held_pose_transport_20260927.scripts',
)


class P4BusinessParameterCapture(BusinessParameterCapture):
    def __init__(self, instance, *, position_reader, public_pose_reader=None,
                 business_modules=BUSINESS_MODULES):
        if instance not in P4_SCRIPTS:
            raise ValueError('Only own current P4-02/03/04')
        super().__init__('S1', position_reader=position_reader,
                         business_modules=business_modules)
        self.instance = instance
        self.public_pose_reader = public_pose_reader
        self.numeric = NumericBoundaryRecorder()

    def _position(self, values):
        sequence, _ = self.position_reader()
        return _count(sequence), 1

    def _pose(self, sequence):
        if self.public_pose_reader is None:
            return None
        return public_pose_reference(self.public_pose_reader(), sequence)

    def _trace(self, frame, event, arg):
        if event != 'call':
            if event == 'return' and id(frame) in self._active:
                row = self._active[id(frame)]
                try:
                    sequence, _ = self._position(frame.f_locals)
                    pose = self._pose(sequence)
                    if pose is not None:
                        row['public_pose_after'] = pose
                except Exception as exc:
                    row['public_pose_after_unavailable'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            return super()._trace(frame, event, arg)
        if frame.f_globals.get('__name__') not in self.modules:
            return None
        function = frame.f_code.co_name
        if function not in ('move', 'hold', 'turn', '_held_pose_move'):
            return None
        if len(self.rows) >= self.max_rows:
            self.truncated = True
            return None
        values = frame.f_locals
        try:
            sequence, index = self._position(values)
            stage_name = 'hold' if function == 'hold' else values['name']
            row = dict(instance=self.instance, source_stage=stage_name,
                       sequence_before=sequence, round=index,
                       original_control_not_inherited=True,
                       original_business_function=function,
                       original_gripper_command=_number(values['grip']))
            if function == 'move':
                # A preceding original inline rotation may have completed
                # since the prior ordinary return. Keep this actual call-time
                # public endpoint before move advances the same Scene; its
                # after pose is not a substitute for this boundary.
                try:
                    pose = self._pose(sequence)
                    if pose is not None:
                        row['public_pose_before'] = pose
                except Exception as exc:
                    row['public_pose_before_unavailable'] = f'{type(exc).__name__}: {str(exc)[:200]}'
                # P402 original move(p,...maxsteps); P403/404 move(target,...steps).
                target_key = 'p' if self.instance == 'P4-02' else 'target'
                bound_key = 'maxsteps' if self.instance == 'P4-02' else 'steps'
                required = (target_key, 'grip', 'clip', 'tol', bound_key, 'stop_harm')
                if not all(k in values for k in required) or type(values['stop_harm']) is not bool:
                    raise ValueError('Exact original P4 move signature missing')
                self.numeric.move(values[target_key], values['grip'], values['clip'],
                                  values['tol'], _count(values[bound_key]))
                args = deepcopy(self.numeric.rows[-1]['arguments'])
                compound_pose = self.instance == 'P4-02' and stage_name == 'naive_real_hot_towel'
                row.update(name=None if compound_pose else 'move_to',
                           arguments={} if compound_pose else args,
                           fixed_numeric_candidate=not compound_pose,
                           original_stop_on_harm=values['stop_harm'])
                if compound_pose:
                    row.update(ordinary_translation_arguments=args,
                        mapping_missing='Original P402 naive terminal move simultaneously sends full-SO3 '
                            'orientation feedback. Translation arguments alone are not that motion; '
                            'the original pose target/path stays unmapped.')
            elif function == 'hold':
                if not all(k in values for k in ('grip', 'n')):
                    raise ValueError('Exact original P4 hold signature missing')
                self.numeric.hold(values['grip'], _count(values['n']))
                row.update(name='set_gripper', arguments=deepcopy(self.numeric.rows[-1]['arguments']),
                           fixed_numeric_candidate=True)
            else:
                row.update(name=None, arguments={}, fixed_numeric_candidate=False,
                    mapping_missing=('Original full-SO3 turn is not an EEF world-z yaw tool; '
                        'observed public endpoint alone does not establish the full pose/path mapping.'
                        if function == 'turn' else
                        'Original held-frame move uses private object-to-EEF relative p/R and full-SO3 '
                        'feedback; body_goal cannot be passed to move_to as EEF xyz.'))
                try:
                    pose = self._pose(sequence)
                    if pose is not None:
                        row['public_pose_before'] = pose
                except Exception as exc:
                    row['public_pose_before_unavailable'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            self.rows.append(row)
            self._active[id(frame)] = row
            return self._trace
        except Exception as exc:
            self.errors.append(f'{function}: {type(exc).__name__}: {str(exc)[:300]}')
            return None

    def __enter__(self):
        if sys.gettrace() is not None or sys.getprofile() is not None:
            raise RuntimeError('Existing debugger/profiler must not be replaced')
        sys.setprofile(self._trace)
        return self

    def __exit__(self, *_):
        sys.setprofile(None)
        self._active.clear()
