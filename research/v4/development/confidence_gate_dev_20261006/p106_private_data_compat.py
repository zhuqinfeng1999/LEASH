"""Use installed MuJoCo field names in the unchanged private P106 scorer.

The old scorer and its geometry/placement predicates stay untouched. This
adapter belongs only to the runner's offline evaluation sink.
"""
from types import SimpleNamespace

from research.v4.candidates.p1_06_exact_shape_scoring_v4_20260926 import scorer as legacy
from research.v4.candidates.p1_06_wine_bucket_t2_20260926 import scorer as original
from research.v4.candidates.p1_ma_fallback_20260925.candidate_v4 import P1V4Evaluator


def exact_shapes(scene):
    sim = scene.backend.sim
    raw = sim.data._data
    # body_xpos/body_xmat are robosuite wrapper aliases, not MjData fields.
    data = SimpleNamespace(body_xpos=raw.xpos, body_xmat=raw.xmat,
                           geom_xpos=raw.geom_xpos, geom_xmat=raw.geom_xmat)
    view = SimpleNamespace(item_name=scene.item_name, target_name=scene.target_name,
        backend=SimpleNamespace(objects_dict=scene.backend.objects_dict,
            sim=SimpleNamespace(model=sim.model, data=SimpleNamespace(_data=data))))
    return legacy.exact_shapes(view)


class P1BucketEvaluator(legacy.P1BucketEvaluator):
    def observe(self, scene):
        self.current_geometry = legacy.apply_exact_fullshape(
            original.bucket_evidence(scene), exact_shapes(scene))
        row = P1V4Evaluator.observe(self, scene)
        row.update(item_on_target=bool(self.current_geometry['in_target']
            and self.current_geometry['floor_supported'] and not row['attached']),
            case_completion=self.current_geometry)
        return row
