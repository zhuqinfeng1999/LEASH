"""Explicit opt-in preparation; no scene, native or model launch authorization."""
from . import binding

INSTANCE = binding.INSTANCE
VERSION = binding.VERSION


def select(instance=INSTANCE, *, method="Ours", version=None, policy=None, calibration=None):
    return binding.selected(instance, method=method, version=version,
                            policy=policy, calibration=calibration)


def prepare(argv, *, method="Ours", version=None, policy=None, calibration=None):
    # The actual original parser is retained; this function never calls main.
    module = select(method=method, version=version, policy=policy, calibration=calibration)
    parsed = module.args(argv)
    if parsed.method != module.__dict__.get("SELECTED_METHOD", parsed.method):
        raise ValueError("actual original parser method differs from selection")
    return module, parsed


def launch(*args, **kwargs):
    raise RuntimeError("E101 SOURCE preparation does not authorize launch")
