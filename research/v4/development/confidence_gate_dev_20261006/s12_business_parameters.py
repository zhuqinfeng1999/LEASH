"""One finite existing-engineering-call parameter capture, without physics startup.

Reads only ordinary numeric arguments at the existing move/hold business
boundary. It changes neither their execution nor return values. No source,
AST, callable, low-level action stream or object graph is persisted. This
scope is engineering-only; its outer metadata must not enter gate evidence.
"""
import json
import math
from pathlib import Path
import sys


BUSINESS_MODULES = {
    'S1': ('research.v4.candidates.s1_v42_tabletop_20260925.scripts_v5_stable_pan300',),
    'S2': ('offline_upper_neck_s2',
           'research.v4.candidates.s2_scene_20260924.v42_physical_preflight',
           'research.v4.candidates.s2_01_upper_neck_122_20260927.probe'),
}


def _number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a motion argument')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite motion argument')
    return value


def _xyz(value):
    # Only this already-consumed three-element numeric argument is read.
    if len(value) != 3:
        raise ValueError('Expected original three-element target')
    return [_number(v) for v in value]


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Expected original nonnegative integer control bound')
    return value


def move_arguments(values):
    """Exact three original local move signatures; no inferred object goal."""
    if all(k in values for k in ('xyz','name','grip','clip','tol','max_steps')):
        target,grip,clip,tol,bound = (values[k] for k in ('xyz','grip','clip','tol','max_steps'))
    elif all(k in values for k in ('target','name','gripper','limit')):
        target,grip,bound = (values[k] for k in ('target','gripper','limit'))
        clip,tol = .025,.012  # Original pill move has these literal parameters.
    elif all(k in values for k in ('target','name','grip','bound','clip','tol')):
        target,grip,clip,tol,bound = (values[k] for k in ('target','grip','clip','tol','bound'))
    else:
        raise ValueError('Original move signature not present')
    return dict(xyz=_xyz(target),gripper=_number(grip),step_clip=_number(clip),
                tol=_number(tol),max_steps=_count(bound),action_scale=.05)


def hold_arguments(values):
    if all(k in values for k in ('grip','steps')):
        grip,count=values['grip'],values['steps']
    elif all(k in values for k in ('gripper','count')):
        grip,count=values['gripper'],values['count']
    else:
        raise ValueError('Original hold signature not present')
    return dict(gripper=_number(grip),steps=_count(count))


def _ordinary_position(values,family):
    scene=values['scene']
    sequence=_count(scene.sequence)
    index=_count(scene.round_index)
    return sequence,index+1 if family=='S1' else index


class BusinessParameterCapture:
    """Call/return argument logging for one Root-owned original probe.

    Only the selected original business function receives local return events;
    no lines or low-level controller calls are recorded. An existing debugger
    is never replaced. Runtime frame/code objects are never saved. The default
    module filter is finite and is not a source-integrity or permission check.
    """
    def __init__(self,family,*,business_modules=None,position_reader=None,max_rows=256):
        if family not in BUSINESS_MODULES:
            raise ValueError('Only fixed S1/S2 engineering probes')
        self.family=family
        self.modules=frozenset(BUSINESS_MODULES[family] if business_modules is None else business_modules)
        if position_reader is not None and not callable(position_reader):
            raise ValueError('Actual scene sequence/round reader must be callable')
        self.position_reader=position_reader
        self.max_rows=_count(max_rows)
        self.rows=[]
        self.errors=[]
        self.truncated=False
        self._active={}

    def _position(self,values):
        if self.position_reader is None:
            return _ordinary_position(values,self.family)
        sequence,index=self.position_reader()
        return _count(sequence),_count(index)+1 if self.family=='S1' else _count(index)

    def _trace(self,frame,event,arg):
        key=id(frame)
        if event=='call':
            if frame.f_globals.get('__name__') not in self.modules:
                return None
            name=frame.f_code.co_name
            if name not in ('move','hold','close_on_handle','wait_for_actual_dose'):
                return None
            if len(self.rows)>=self.max_rows:
                self.truncated=True
                return None
            values=frame.f_locals
            try:
                sequence,round_index=self._position(values)
                row=dict(sequence_before=sequence,round=round_index,
                         source_stage=values['name'] if name=='move' else name)
                if name=='move':
                    row.update(name='move_to',arguments=move_arguments(values),
                        fixed_numeric_candidate=True,original_control_not_inherited=True)
                elif name=='hold':
                    row.update(name='set_gripper',arguments=hold_arguments(values),
                        fixed_numeric_candidate=True)
                else:
                    row.update(name=None,arguments={},fixed_numeric_candidate=False,
                        source_stage=values['stage'] if name=='close_on_handle' else name,
                        mapping_missing=('165-control moving target/CLOSE followed by12closed controls; '
                            'do not rename as set_gripper' if name=='close_on_handle' else
                            'private dose-dependent early stop within60OPENcontrols; '
                            'not an online evaluator condition'))
            except Exception as exc:
                self.errors.append(f'{name}: {type(exc).__name__}: {str(exc)[:300]}')
                return None
            self.rows.append(row)
            self._active[key]=row
            return self._trace
        if key not in self._active:
            return None
        if event=='exception':
            self._active[key]['exception_observed']=type(arg[1]).__name__
        elif event=='return':
            row=self._active.pop(key)
            try:
                sequence,round_index=self._position(frame.f_locals)
                row.update(sequence_after=sequence,round_after=round_index,
                           actual_controls=sequence-row['sequence_before'])
            except Exception as exc:
                self.errors.append(f'return: {type(exc).__name__}: {str(exc)[:300]}')
            return None
        return self._trace

    def __enter__(self):
        if sys.gettrace() is not None:
            raise RuntimeError('Existing debugger/call trace cannot be replaced')
        sys.settrace(self._trace)
        return self

    def __exit__(self,*_):
        sys.settrace(None)
        # Keep finite rows/errors; do not keep runtime frames or callables.
        self._active.clear()

    def save(self,path):
        path=Path(path)
        with path.open('x') as stream:
            for row in self.rows:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        return dict(path=str(path),calls=len(self.rows),
                    truncated=self.truncated,recording_errors=list(self.errors))


def fixed_public_contact_candidate(public_contact_eef_xyz,*,public_reference):
    """Concrete ordinary-tool test, requiring a public-derived EEF contact pose.

    A raw visible-surface point is insufficient. Caller must derive the pose
    from current public RGB-D and documented pad geometry; this helper neither
    selects the pose nor reads a hidden grasp zone. No automatic insertion.
    This is a different, unqualified fixed-contact engineering candidate, not
    the original adaptive moving-CLOSE and not a planner demonstration.
    """
    required=('episode','frame','view','eef_reference','pad_geometry_source')
    if not all(k in public_reference and public_reference[k] is not None for k in required):
        raise ValueError('Current public contact-pose provenance incomplete')
    if public_reference['eef_reference']!='documented_grip_site':
        raise ValueError('Surface point cannot be substituted for a contact EEF pose')
    return dict(engineering_only=True,accepted=False,
        source='current public contact pose + robot pad geometry',
        public_reference=dict(public_reference),
        stages=[dict(name='move_to',arguments=dict(xyz=_xyz(public_contact_eef_xyz),
                     gripper=-1.,step_clip=.003,tol=.009,max_steps=165,action_scale=.05)),
                dict(name='set_gripper',arguments=dict(gripper=1.,steps=12))],
        distinction='Fixed open-gripper positioning then12CLOSE; differs from original moving-CLOSE, '
            'no feedback following object motion and no early private attachment stop.',
        stop='After12CLOSE; evaluate real contact/attachment offline. '
            'A later public verification lift is a separately explicit ordinary candidate, '
            'not a success inferred from CLOSE.')
