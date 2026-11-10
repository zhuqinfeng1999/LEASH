"""Own container binding; reuse the original unknown-tilt outlet without an estimator."""
from functools import lru_cache
from types import SimpleNamespace
from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import components as original_components
from research.v4.accelerated.p3_gated_adapter import P3GatedAdapter

IDS=('E1-05','E1-06')
@lru_cache(maxsize=2)
def components(instance):
    if instance not in IDS:raise ValueError('This candidate binds only E105/E106 containers')
    old=original_components(instance)
    if old.binding.category!='container_unknown_contents':raise ValueError('Unexpected category')
    fields=dict(vars(old))
    fields.update(E1GatedAdapter=P3GatedAdapter,cpu_call_path_ready=True,
        native_mask_review_ready=False,method_launch_authorized=False,pending_reason=None,
        tilt_protocol='Held container initial tilt remains unknown; any held rotational command BLOCKs before controller submission. Unrotated translations remain permitted; releases use fresh evidence review.')
    return SimpleNamespace(**fields)
