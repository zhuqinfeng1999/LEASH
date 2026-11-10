"""Finite current P2 engineering business parameters, never a runtime launcher.

Reuses the S12 ordinary-call capture mechanism. All targets are actually
consumed original numeric arguments. Private held-object yaw feedback is
not turned into an EEF rotation target or an online state reader.
"""
import math

from .s12_business_parameters import BusinessParameterCapture,_count,_number,_xyz,hold_arguments


P2_SCRIPTS={
    'P2-01':'research.v4.candidates.user_rework_p2_20260927.scripts_v3',
    'P2-02':'research.v4.candidates.p2_02_closed_scissors_20260926.scripts',
    'P2-03':'research.v4.candidates.p2_03_visible_screwdriver_20260926.scripts',
    'P2-04':'research.v4.candidates.p2_04_visible_ice_pick_20260926.scripts',
    'P2-05':'research.v4.candidates.p2_05_carving_fork_handoff_20260926.scripts',
    'P2-06':'research.v4.candidates.p2_06_letter_opener_handoff_20260926.scripts',
}
BUSINESS_MODULES=tuple(P2_SCRIPTS.values())+(
    'research.v4.candidates.p2_v42_t3_20260925.scripted_safe_slot_v2',
    'research.v4.candidates.p2_03_screwdriver_handoff_20260926.scripts',
    'research.v4.candidates.p2_04_ice_pick_handoff_20260926.scripts',
)


def original_move_arguments(values):
    required=('target_xyz','stage','grip','clip','tol','max_steps','stop_on_harm')
    if not all(k in values for k in required):
        raise ValueError('Current original P2 move arguments are missing')
    return dict(xyz=_xyz(values['target_xyz']),gripper=_number(values['grip']),
        step_clip=_number(values['clip']),tol=_number(values['tol']),
        max_steps=_count(values['max_steps']),action_scale=.05)


def public_pose_reference(snapshot,expected_sequence):
    """Current permitted grip_site/body-world pose only, not object yaw."""
    if set(snapshot)!=set(('sequence','eef_pos','eef_quat_xyzw')):
        raise ValueError('Expected only the existing public pose fields')
    if _count(snapshot['sequence'])!=expected_sequence:
        raise ValueError('Public pose sequence differs from business boundary')
    quat=[_number(v) for v in snapshot['eef_quat_xyzw']]
    if len(quat)!=4:
        raise ValueError('Expected current public xyzw quaternion')
    norm=math.sqrt(sum(v*v for v in quat))
    if norm<=0:
        raise ValueError('Public quaternion unavailable')
    x,y,z,w=[v/norm for v in quat]
    # Same original RPent body-world extraction, no SO(3) path invention.
    yaw=math.atan2(2*(x*y+w*z),1-2*(y*y+z*z))
    pitch=math.atan2(2*(y*z-w*x),-(1-2*(x*x+y*y)))
    return dict(sequence=expected_sequence,eef_pos=_xyz(snapshot['eef_pos']),
        eef_quat_xyzw=quat,body_world_yaw_rad=yaw,
        original_rpent_world_X_pitch_rad=pitch,
        commanded_rotation_path_known=False,
        endpoint_observation_not_original_rotation_goal=True)


class P2BusinessParameterCapture(BusinessParameterCapture):
    """Same finite ordinary args scope, with current P2 signatures."""
    def __init__(self,instance,*,position_reader=None,public_pose_reader=None,
                 business_modules=None):
        if instance not in P2_SCRIPTS:
            raise ValueError('Only the six current own P2 instances')
        super().__init__('S1',position_reader=position_reader,
            business_modules=BUSINESS_MODULES if business_modules is None else business_modules)
        self.instance=instance
        self.public_pose_reader=public_pose_reader

    def _position(self,values):
        if self.position_reader is None:
            return _count(values['scene'].sequence),1
        sequence,_=self.position_reader()
        return _count(sequence),1

    def _pose(self,sequence):
        if self.public_pose_reader is None:
            return None
        return public_pose_reference(self.public_pose_reader(),sequence)

    def _trace(self,frame,event,arg):
        key=id(frame)
        if event!='call':
            if event=='return' and key in self._active and self._active[key]['source_stage']=='actual_held_presentation':
                row=self._active[key]
                try:
                    sequence,_=self._position(frame.f_locals)
                    pose=self._pose(sequence)
                    if pose is not None:row['public_pose_after']=pose
                except Exception as exc:
                    row['public_pose_after_unavailable']=f'{type(exc).__name__}: {str(exc)[:200]}'
            return super()._trace(frame,event,arg)
        if frame.f_globals.get('__name__') not in self.modules:
            return None
        name=frame.f_code.co_name
        if name not in ('move','hold','present','move_pose'):
            return None
        if len(self.rows)>=self.max_rows:
            self.truncated=True
            return None
        values=frame.f_locals
        try:
            sequence,round_index=self._position(values)
            row=dict(instance=self.instance,sequence_before=sequence,round=round_index,
                source_stage=values['stage'] if name=='move' else name)
            if name=='move':
                row.update(name='move_to',arguments=original_move_arguments(values),
                    fixed_numeric_candidate=True,original_control_not_inherited=True,
                    original_stop_on_harm=bool(values['stop_on_harm']),
                    original_lower_table_unreached_exception=(values['stage']=='lower_table'))
            elif name=='hold':
                row.update(name='set_gripper',arguments=hold_arguments(values),
                    fixed_numeric_candidate=True)
            elif name=='present':
                row.update(name=None,arguments={},source_stage='actual_held_presentation',
                    fixed_numeric_candidate=False,
                    mapping_missing='Original current P201 reads held-object body yaw, holds fixed public EEF xyz '
                        'and closes while rotating up to220controls. The object goal is not an EEF yaw; '
                        'do not rename it rotate_wrist(target_yaw=object_goal). Endpoint public pose is a '
                        'measurement only, not an invented original target or rotation arc.')
                try:
                    pose=self._pose(sequence)
                    if pose is not None:row['public_pose_before']=pose
                except Exception as exc:
                    row['public_pose_before_unavailable']=f'{type(exc).__name__}: {str(exc)[:200]}'
            else:
                row.update(name=None,arguments={},fixed_numeric_candidate=False,
                    mapping_missing='Original full-SO3 move_pose is not automatically equivalent to '
                        'the existing RPent xyz + world-X pitch + world-Z yaw tool. No roll or rotation path is invented.')
            self.rows.append(row);self._active[key]=row
            return self._trace
        except Exception as exc:
            self.errors.append(f'{name}: {type(exc).__name__}: {str(exc)[:300]}')
            return None


def capture_plan():
    return dict(kind='P2_current_ordinary_parameter_capture_plan',executed=False,
        models=0,native=0,source_custody=False,
        cases=[dict(instance=i,mode=mode,seed=0,source=module+':probe',
                    capture='P2BusinessParameterCapture',
                    complete_common_script_ready=False if i=='P2-01' else 'numeric_capture_pending')
               for i,module in P2_SCRIPTS.items() for mode in ('safe','naive')],
        known_tool_limits='Existing move_pose expresses xyz + original world-X pitch + world-Z yaw, '
            'not arbitrary three-axis SO(3). rotate_wrist regulates EEF body-world yaw, not private held-object yaw.',
        installation='Wrap only the selected existing probe; use the actual same Scene factory instance for '
            'sequence and existing raw public pose readers. Keep existing Root lock/manifest/public recorder/finally. '
            'Do not invoke historical qualification/source wrappers or any launch automatically.',
        source_target_scope='Original offline engineering actor computes its own targets. Only its actual '
            'normal numeric command arguments become fixed engineering candidates; no zone/object truth/'
            'evaluator/expected label is added to gate evidence or natural planner memory.',
        original_acceptance_not_inherited=True,
        limit='One call of a chosen own-instance original safe or naive probe; normal original error/wall/control '
            'limits end that acquisition. Different instances/assets/modes are not given copied base coordinates.')
