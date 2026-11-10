"""Whole selected old plus own current source admission; no readiness override."""
import hashlib,json
from pathlib import Path
from research.v4.candidates.r201_current_t1_hd_method_20260928 import binding as parent
from research.v4.candidates.user_rework_r5_food_bindings_20260927 import entry as meal_entry

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def current():
    pins=dict(parent.admit())
    # The six canonical sidecars were intentionally replaced at publication.
    # Keep the old receipt immutable and prove no runtime/asset field drift.
    old=json.loads((meal_entry.HERE/'source_freeze.json').read_text())
    meal=meal_entry.hashes(require_freeze=False)
    sidecars={f'research/v4/review_videos/R5-{i:02d}.json' for i in range(1,7)}
    if old.keys()!=meal.keys() or any(old[k]!=meal[k] for k in old if k not in sidecars):
        raise RuntimeError('Own meal source/asset changed beyond published sidecars')
    pins.update(meal)
    files=[*HERE.glob('*.py'),
           ROOT/'research/v4/candidates/r5_derived_natural_method_20260926/offline_evaluator.py',
           ROOT/'research/v4/candidates/e2_laundry_method_20260926/public_intent.py',
           ROOT/'research/v4/configs/leash_safety_policy_draft_agecues_20260925.md',
           meal_entry.HERE/'source_freeze.json',HERE/'meal_historical_guard_readonly.json']
    files.extend(ROOT/f'research/v4/results/r5_{i:02d}_user_revision_v42_instance_acceptance_20260928.json' for i in range(1,7))
    pins.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return dict(sorted(pins.items()))
def admit():
    actual=current();expected=json.loads((HERE/'source_freeze.json').read_text())
    if actual!=expected:raise RuntimeError('Current own R5 REAL source admission mismatch')
    return actual
