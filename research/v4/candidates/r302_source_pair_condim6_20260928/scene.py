"""Exactly two condim fields, unchanged own fixed80 constructor sequence."""
import hashlib
import json
from pathlib import Path
from types import FunctionType

import mujoco
import numpy as np

from research.v4.candidates.r302_front_support_classification_v3_20260928 import scene as old
from .admit import verify
from .support import PARAMS, allowed_pairs, sample

H = Path(__file__).resolve().parent


class SixDimSourcePairEnv(old.previous.pair.SourcePairFrontShortVEnv):
    def _load_model(self):
        super()._load_model()
        pairs = list(self.model.root.find('contact').findall('pair'))
        expected = {('r302_source_pair_' + r['rail'].rsplit('_', 1)[-1]): r for r in PARAMS['rows']}
        if len(pairs) != 2 or {p.get('name') for p in pairs} != set(expected):
            raise RuntimeError('Exact precompile two support pairs changed')
        for p in pairs:
            row = expected[p.get('name')]
            if p.get('condim') != '3' or (p.get('geom1'), p.get('geom2')) != (row['can'], row['rail']):
                raise RuntimeError('Exact original pair3 identity changed')
            p.set('condim', '6')


_RAISED = FunctionType(old._RAISED.__code__, dict(old._RAISED.__globals__, H=H), closure=old._RAISED.__closure__)
_FRONT = FunctionType(old._FRONT.__code__, dict(old._FRONT.__globals__, H=H,
                      _RAISED_INIT=_RAISED, FrontShortVEnv=SixDimSourcePairEnv), closure=old._FRONT.__closure__)


def selected_front_init(self, *args, **kwargs):
    _FRONT(self, *args, **kwargs)
    m = self.backend.sim.model._model
    before = mujoco.MjModel.from_binary_path(str(old.H / 'startup_seed0/compiled_model.mjb'))
    target = H / 'startup_seed0/compiled_model.mjb'
    mujoco.mj_saveModel(m, str(target))
    same, changed = [], []
    for name in dir(m):
        a, b = getattr(before, name, None), getattr(m, name, None)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            equal = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
            (same if equal else changed).append(name)
    pairs = allowed_pairs(m)
    before_names = [(i, before.geom(i).name) for i in range(before.ngeom)]
    after_names = [(i, m.geom(i).name) for i in range(m.ngeom)]
    dimensions = {n: (int(getattr(before, n)), int(getattr(m, n)))
                  for n in ('nq', 'nv', 'nbody', 'njnt', 'ngeom', 'nsite', 'ncam', 'nu', 'nsensor', 'neq', 'npair')}
    passed = bool(changed == ['pair_dim'] and np.array_equal(before.pair_dim, [3, 3])
                  and np.array_equal(m.pair_dim, [6, 6]) and before.names == m.names
                  and before_names == after_names and all(a == b for a, b in dimensions.values()))
    receipt = {'passed': passed, 'before_any_startup_controls': True,
               'changed_model_arrays': changed, 'same_model_array_count': len(same),
               'dimension_checks': dimensions, 'old_name_buffer_exact': before.names == m.names,
               'old_geom_ID_name_exact': before_names == after_names,
               'old_pair_dim': before.pair_dim.tolist(), 'new_pair_dim': m.pair_dim.tolist(),
               'compiled_pair_metadata': [dict(v, geom_ID_set=sorted(k)) for k, v in pairs.items()],
               'new_physical_version_not_equivalent': True,
               'compiled_model_before_startup_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (H / 'compiled_condim_guard.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not passed:
        raise RuntimeError('First unexplained model change beyond exact pair_dim3 to6')


_INIT = FunctionType(old._INIT.__code__, dict(old._INIT.__globals__, H=H,
                     _FRONT_INIT=selected_front_init, sample=sample, verify=verify), closure=old._INIT.__closure__)


class SixDimSourceSupportCanScene(old.SourceSupportClassificationV3CanScene):
    __init__ = _INIT
    layout_version = 'R302-same-source-fixed80-two-real-support-pairs-condim6-v1'
