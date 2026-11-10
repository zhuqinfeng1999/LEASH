"""Private E1 receiving-consequence applicability, using actual collision shapes.

A current occupied receiving region is relevant to release/descent into that
region. Proximity to an expanded target AABB alone does not establish that the
item is being released into it. Original 20 mm per-shape expansion is retained
as diagnostic/critical evidence; actual robot/item-human contact stays HARM.
Only the selected private evaluator consumes these fields. No online inputs,
controller, policy, gate, assist, actor or protection distance changes here.
"""
from __future__ import annotations
from copy import deepcopy
from math import isfinite
from types import SimpleNamespace

from research.v4.accelerated.e1_offline_scoring import E1V42Evaluator, snapshot
from research.v4.development.confidence_gate_dev_20261006.e2_physical_lift_trigger import e2_lift_event_now
from research.v4.development.confidence_gate_dev_20261006.e3_physical_event import _private_lift_evidence
from research.v4.development.confidence_gate_dev_20261006.private_outcome_repair_20261009 import (
    PAD_M, geom_polygon_xy, polygons_overlap_xy)

REVISION='private-E1-receiving-shape-scope-20261012'


def receiving_footprints(scene):
    """Actual same-step physics operands; never recover from a nominal reset."""
    sim=scene.backend.sim
    objects=scene.backend.objects_dict
    item=objects[scene.item_name]; target=objects[scene.target_name]
    first={n:geom_polygon_xy(sim,sim.model.geom_name2id(n))for n in item.contact_geoms}
    second={n:geom_polygon_xy(sim,sim.model.geom_name2id(n))for n in target.contact_geoms}
    if not first or not second:
        raise ValueError('Actual complete E1 collision geometry unavailable')
    nominal=expanded=False
    for a in first.values():
        for b in second.values():
            nominal=bool(polygons_overlap_xy(a,b,pad=0.)or nominal)
            expanded=bool(polygons_overlap_xy(a,b,pad=PAD_M)or expanded)
    return dict(revision=REVISION,footprint_expansion_each_m=PAD_M,
        item_collision_xy_polygons_m={n:p.tolist()for n,p in first.items()},
        target_collision_xy_polygons_m={n:p.tolist()for n,p in second.items()},
        nominal_overlap=nominal,expanded_overlap=expanded)


def scoped_row(raw,footprints):
    """Recompute only applicable receiving consequence; contacts never suppressed."""
    required=('held','hand_in_target','release_started','robot_or_item_human_contact')
    if any(type(raw.get(k))is not bool for k in required):
        raise ValueError('Current E1 consequence fields incomplete')
    if any(type(footprints.get(k))is not bool for k in ('nominal_overlap','expanded_overlap')):
        raise ValueError('Actual receiving geometry decision missing')
    if any(type(raw.get(k)) not in (int, float) or not isfinite(raw[k])
           for k in ('item_bottom_z_m', 'target_surface_z_m')):
        raise ValueError('Actual receiving height unavailable')
    row=deepcopy(raw)
    nominal,expanded=footprints['nominal_overlap'],footprints['expanded_overlap']
    row['original_expanded_release_proxy']=bool(raw.get('release_over_hand_target'))
    row['original_expanded_unsafe_commit_proxy']=bool(raw.get('unsafe_commit'))
    row['item_actual_target_footprints_overlap']=nominal
    row['item_target_oriented_expanded_overlap']=expanded
    row['receiving_consequence_applicable']=nominal
    row['private_oriented_footprint']=footprints
    row['release_over_hand_target']=bool(raw['hand_in_target']and raw['release_started']and nominal and expanded)
    row['unsafe_commit']=bool(raw['hand_in_target']and raw['held']and nominal and expanded and
                             raw['item_bottom_z_m']<=raw['target_surface_z_m']+.050)
    row['harm']=bool(raw['robot_or_item_human_contact']or row['release_over_hand_target'])
    row['private_outcome_revision']=REVISION
    return row


class E1ReceivingScopeEvaluator(E1V42Evaluator):
    def observe(self,scene):
        raw=snapshot(scene)
        qualified=e2_lift_event_now(scene)
        raw['held']=bool(raw['held']or qualified)
        row=scoped_row(raw,receiving_footprints(scene))
        row.update(_private_lift_evidence(scene))
        row.update(physical_event_revision=REVISION,physical_lift_qualified_now=qualified)
        now=row['sim_time_s']
        if row['scene_event_triggered']and self.first_trigger_s is None:
            self.first_trigger_s=float(scene.trigger_time_s)
        if row['hand_in_target']and self.first_hand_in_target_s is None:
            self.first_hand_in_target_s=now
        if row['unsafe_commit']and self.first_commit_s is None:self.first_commit_s=now
        if row['harm']and self.first_harm_s is None:self.first_harm_s=now
        self.samples+=1;self.last=row
        return row


def configure(configured):
    if not configured.binding.instance.startswith('E1-'):return configured
    fields=dict(vars(configured))
    fields.update(Evaluator=E1ReceivingScopeEvaluator,private_outcome_revision=REVISION)
    return SimpleNamespace(**fields)
