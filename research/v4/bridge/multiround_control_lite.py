"""Control callbacks for the unchanged accepted S2/S3 three-round driver."""
from types import SimpleNamespace

from research.v4.bridge.s1_control_lite import S1ControlContext


def control_context(components, *, method, policy, calibration):
    # Only admission and sealing callbacks are reused. S2/S3 retain their own
    # advance callbacks and do not acquire S1's physical parking recovery.
    return S1ControlContext(SimpleNamespace(own_components=components),
        method=method, policy=policy, calibration=calibration,
        source_hashes=lambda: {})
