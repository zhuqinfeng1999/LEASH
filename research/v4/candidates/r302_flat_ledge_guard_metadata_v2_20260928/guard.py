"""Two source-explained metadata corrections; physical guard stays strict."""
import inspect
from types import FunctionType

import numpy as np
from research.v4.candidates.r302_flat_ledge_root_qualifier_20260928 import guard as parent
from .math_reference import reference


def compare(actual,seed):
    expected,receipt,paths=reference(seed);wood=expected.body('r302_v_wood_cradle').id
    source=inspect.getsource(parent.compare)
    before="'nnames':old.nnames+name_delta,'nnames_map':old.nnames_map,'nbuffer':old.nbuffer}"
    after="'nnames':old.nnames+name_delta,'nnames_map':old.nnames_map,'nbuffer':old.nbuffer+len(EXPECTED_PATHS)-old.npaths,'npaths':len(EXPECTED_PATHS)}"
    if source.count(before)!=1:raise RuntimeError('Exact allocation-only guard seam changed')
    selected=source.replace(before,after,1)
    if selected.replace(after,before,1)!=source:raise RuntimeError('Unexpected guard algorithm change')
    ns=dict(parent.compare.__globals__,reference=lambda s:(expected,receipt),
        support_inertia=lambda:(0.,expected.body_pos[wood].copy(),np.zeros((3,3))),EXPECTED_PATHS=paths)
    exec(compile(selected,__file__+'::only-path-allocator-delta','exec'),ns)
    result=ns['compare'](actual,seed)
    checks=dict(raw_path_buffer_exact=actual.paths==paths,
        fixed_wood_zero_mass=float(actual.body_mass[wood])==0.,
        fixed_wood_zero_inertia=parent.same(actual.body_inertia[wood],np.zeros(3)),
        fixed_wood_compiler_default_ipos=parent.same(actual.body_ipos[wood],expected.body_pos[wood]),
        fixed_wood_compiler_default_iquat=parent.same(actual.body_iquat[wood],expected.body_quat[wood]),
        fixed_wood_has_no_joint=int(actual.body_jntnum[wood])==0,
        exact_decoded_asset_file_hashes=all(x['old_sha256']==x['new_sha256']for x in receipt['path_rows']),
        unchanged_guard_inverse_source_exact=selected.replace(after,before,1)==source)
    result.update(v2_checks=checks,passed=result['passed']and all(checks.values()),
        old840_failure_immutable=True,metadata_version='v2-actual-group0-static-default-and-exact-path-repacking',
        support_not_measured=True,no_contacts_exempted=True,no_constructor_or_controls=True)
    if not all(checks.values()):result['errors']+=['v2.'+k for k,v in checks.items()if not v]
    return result
