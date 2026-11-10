"""Finite ordinary P104/105/106 parameter capture; no execution at import.

Private object-centering is not converted into an equivalent ordinary tool.
Only actual numeric business arguments and permitted EEF return positions
are retained. Existing call/return recorder owns cleanup and row bounds.
"""
from .s12_business_parameters import BusinessParameterCapture,_count,_number,_xyz,hold_arguments

BUSINESS_MODULES=(
    'research.v4.candidates.p1_ma_fallback_20260925.candidate',
    'research.v4.candidates.p1_05_exact_clearance_v3_20260926.script',
    'research.v4.candidates.p1_06_reachable_route_v3a_20260926.script',
)


def original_move(values,instance):
    required=('target_xyz','stage','grip','clip','tol','max_steps','stop_on_harm')
    if not all(k in values for k in required):
        raise ValueError('Original P1 ordinary move arguments missing')
    clip=_number(values['clip'])
    # Exact original override runs inside move, after its call event.
    if instance in ('P1-05','P1-06') and _number(values['grip'])>0 and values['stage']!='lower_plate':
        clip=.030
    return dict(xyz=_xyz(values['target_xyz']),gripper=_number(values['grip']),
        step_clip=clip,tol=_number(values['tol']),max_steps=_count(values['max_steps']),action_scale=.05)


class P1OwnBusinessParameterCapture(BusinessParameterCapture):
    def __init__(self,instance,*,position_reader,public_position_reader,business_modules=None):
        if instance not in ('P1-04','P1-05','P1-06'):
            raise ValueError('Only three own current P1 bindings')
        super().__init__('S1',position_reader=position_reader,
            business_modules=BUSINESS_MODULES if business_modules is None else business_modules)
        self.instance=instance
        self.public_position_reader=public_position_reader

    def _position(self,values):
        seq,_=self.position_reader()
        return _count(seq),1

    def _public_position(self,seq):
        actual=self.public_position_reader()
        if set(actual)!=set(('sequence','eef_pos')) or _count(actual['sequence'])!=seq:
            raise ValueError('Public EEF position must be same business boundary')
        return dict(sequence=seq,eef_pos=_xyz(actual['eef_pos']),
            source='actual raw robot0_eef_pos at original business boundary')

    def _trace(self,frame,event,arg):
        key=id(frame)
        if event!='call':
            if event=='return' and key in self._active:
                row=self._active[key]
                try:
                    seq,_=self._position(frame.f_locals)
                    row['public_position_after']=self._public_position(seq)
                except Exception as exc:
                    row['public_position_after_unavailable']=f'{type(exc).__name__}: {str(exc)[:200]}'
            return super()._trace(frame,event,arg)
        if frame.f_globals.get('__name__') not in self.modules or frame.f_code.co_name not in ('move','hold'):
            return None
        if len(self.rows)>=self.max_rows:
            self.truncated=True
            return None
        values=frame.f_locals;name=frame.f_code.co_name
        try:
            seq,index=self._position(values)
            row=dict(instance=self.instance,round=index,sequence_before=seq,
                source_stage=values['stage'] if name=='move' else name,
                public_position_before=self._public_position(seq))
            if name=='move':
                dynamic=bool(values.get('center_payload',False))
                row.update(name=None if dynamic else 'move_to',
                    arguments={} if dynamic else original_move(values,self.instance),
                    original_numeric_arguments=original_move(values,self.instance),
                    fixed_numeric_candidate=not dynamic,
                    original_control_not_inherited=True,
                    original_stop_on_harm=bool(values['stop_on_harm']),
                    original_private_center_payload=dynamic,
                    original_lower_support_unreached_exception=values['stage']=='lower_plate')
                if dynamic:
                    row['mapping_missing']='Per-control private held-object/target XY feedback; the numeric call goal is not its executed fixed target.'
            else:
                row.update(name='set_gripper',arguments=hold_arguments(values),fixed_numeric_candidate=True)
            self.rows.append(row);self._active[key]=row
            return self._trace
        except Exception as exc:
            self.errors.append(f'{name}: {type(exc).__name__}: {str(exc)[:300]}')
            return None

