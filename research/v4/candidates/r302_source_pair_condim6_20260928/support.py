"""Same exact support classifier, plus raw new contact constraint evidence."""
import copy
from pathlib import Path
from types import FunctionType

import mujoco
import numpy as np

from research.v4.candidates.r302_front_support_classification_v3_20260928 import support as old

H = Path(__file__).resolve().parent
PARAMS = copy.deepcopy(old.PARAMS)
for row in PARAMS['rows']:
    if row['dim'] != 3:
        raise RuntimeError('Frozen original pair dimension changed')
    row['dim'] = 6

allowed_pairs = FunctionType(old.allowed_pairs.__code__, dict(old.allowed_pairs.__globals__, PARAMS=PARAMS),
                             closure=old.allowed_pairs.__closure__)
classified_contacts = FunctionType(old.classified_contacts.__code__,
                                   dict(old.classified_contacts.__globals__, allowed_pairs=allowed_pairs),
                                   closure=old.classified_contacts.__closure__)
_ENRICH = FunctionType(old.enrich.__code__, dict(old.enrich.__globals__, allowed_pairs=allowed_pairs),
                      closure=old.enrich.__closure__)


def enrich(scene, row):
    _ENRICH(scene, row)
    m, d = scene.backend.sim.model._model, scene.backend.sim.data._data
    contact_types = tuple(int(getattr(mujoco.mjtConstraint, n)) for n in (
        'mjCNSTR_CONTACT_FRICTIONLESS', 'mjCNSTR_CONTACT_PYRAMIDAL', 'mjCNSTR_CONTACT_ELLIPTIC'))
    row['actual_solver_nefc'] = int(d.nefc)
    row['actual_solver_cone_enum'] = int(m.opt.cone)
    for record in row['actual_source_support_contacts']:
        index = record['contact_id']
        contact = d.contact[index]
        indices = np.flatnonzero((np.asarray(d.efc_id[:d.nefc]) == index)
                                 & np.isin(np.asarray(d.efc_type[:d.nefc]), contact_types))
        record.update(actual_contact_efc_address=int(contact.efc_address),
                      actual_contact_constraint_indices=indices.tolist(),
                      actual_contact_constraint_types=d.efc_type[indices].tolist(),
                      actual_contact_constraint_force=d.efc_force[indices].tolist(),
                      physical_contact_dimension_before=3, physical_contact_dimension_now=6,
                      newly_enabled_dimensions=['torsion_about_normal', 'rolling_tangent1', 'rolling_tangent2'],
                      friction_coefficients_not_changed=True)


sample = FunctionType(old.sample.__code__, dict(old.sample.__globals__, H=H,
                      exact_contacts=classified_contacts, enrich=enrich), closure=old.sample.__closure__)
EXECUTED_SOURCE = old.EXECUTED_SOURCE
