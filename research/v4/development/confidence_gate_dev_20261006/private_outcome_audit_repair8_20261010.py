"""Evidenced private outcome revisions; raw episodes and controls are immutable.

Only complete original per-control operands may be recomputed. Missing event
instantiation or P1 route inputs are engineering exclusions, never fabricated
successful trajectories. This module is never an online evidence source.
"""
from hashlib import sha256
import json
from pathlib import Path

REVISION='private-physical-event-and-support-audit-repair8-20261010'


def configure(configured):
    family=configured.binding.instance.split('-')[0]
    if family=='P1':
        from .p1_physical_completion_repair8 import configure as physical
        configured=physical(configured)
    if family=='P4':
        from .p4_all_physical_completion_repair8 import configure as physical_hot_tool
        configured=physical_hot_tool(configured)
    if family in ('E1','E2','E3','S1'):
        from .e_s_outcome_repair8_20261010 import configure as event
        configured=event(configured)
    if family in ('P1','P4','E1','E2','E3','S1') or (family.startswith('I') and not configured.binding.instance.endswith('-01')):
        configured.private_outcome_revision=REVISION
    return configured


def p402_trace(rows,episode):
    from .p4_heatmat_support_repair8 import heatmat_certificate
    from research.v4.candidates.p4_support_scoring_20260926.evaluator import P4SupportEvaluator
    if not rows:raise ValueError('Complete P4 trace required')
    evaluator=P4SupportEvaluator();corrected=[]
    initial=rows[0]['item_bottom_z_m']
    for before in rows:
        fields=('physical_contacts','item_com_xyz_m','target_support_region_xy_m',
                'assisted_attached','iron_linear_speed_m_s','iron_angular_speed_rad_s','hot_shirt_contact')
        if any(k not in before for k in fields):raise ValueError('Actual P4 contact/support operands missing')
        row=dict(before,**heatmat_certificate(contacts=before['physical_contacts'],
            com_xy=before['item_com_xyz_m'][:2],target_polygon=before['target_support_region_xy_m'],
            attached=before['assisted_attached'],linear_speed=before['iron_linear_speed_m_s'],
            angular_speed=before['iron_angular_speed_rad_s'],hot_cloth=before['hot_shirt_contact']))
        # Saved original attachment-to-detachment transition, not a proposed
        # OPEN or a success inferred from the final image.
        release=(before.get('detached_in_stand_s') is not None and
                 abs(before['detached_in_stand_s']-before['sim_time_s'])<1e-9)
        corrected.append(evaluator.observe_row(row,release_started=release,initial_bottom_z=initial))
    result=evaluator.finish((episode.get('terminal')or{}).get('status'),
                            failure=(episode.get('physical')or{}).get('failure'))
    return corrected,result['label'],result['completion_evidence']


def derive(instance,rows,episode):
    from .e_s_outcome_repair8_20261010 import correct_e2_trace,correct_s1_trace
    if instance.startswith('E2-'):
        return correct_e2_trace(rows,failure=(episode.get('physical')or{}).get('failure'))
    if instance.startswith('S1-'):
        return correct_s1_trace(rows)
    if instance=='P4-02':
        return p402_trace(rows,episode)
    raise ValueError('Instance outside repair8 complete-trace correction')


def validate_corrected_trace(episode,output,correction):
    if correction.get('evaluation_revision')!=REVISION:raise ValueError('Unknown audit scoring revision')
    raw_path=output/'evaluator-only.jsonl'
    if sha256(raw_path.read_bytes()).hexdigest()!=correction.get('raw_evaluator_sha256'):
        raise ValueError('Corrected finding does not match raw physical trace')
    name=correction.get('corrected_evaluator_file')
    if not name or Path(name).name!=name:raise ValueError('Corrected trace must be episode local')
    target=output/name
    if sha256(target.read_bytes()).hexdigest()!=correction.get('corrected_evaluator_sha256'):
        raise ValueError('Corrected physical trace hash mismatch')
    raw=[json.loads(x)for x in raw_path.read_text().splitlines()]
    saved=[json.loads(x)for x in target.read_text().splitlines()]
    if not raw or len(raw)!=len(saved):raise ValueError('Complete actual trace required')
    for before,after in zip(raw,saved):
        if before['sequence']!=after['sequence'] or before['sim_time_s']!=after['sim_time_s']:
            raise ValueError('Actual execution sequence/time association changed')
    expected,label,_=derive(episode['instance'],raw,episode)
    if expected!=saved or label!=correction['label']:
        raise ValueError('Derived finding does not match deterministic complete-trace revision')
    return target
