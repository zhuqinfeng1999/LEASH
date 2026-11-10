"""Read the unchanged handoff G3 text as UTF-8 on Windows too.

Keep the frozen hardware-prompt function and its version. Only its scoped
text reader is instance-local; no process locale or shared module is patched.
"""
from pathlib import Path
from types import FunctionType

from research.v4.development.decision_gate_dev_20261019 import handoff
from research.v4.development.g3_transition_dev_20261019 import reviewer


def utf8_scoped_prompt(gate, *args, **kwargs):
    if gate == 'G3':
        return Path(reviewer.__file__).with_name('prompt_G3.txt').read_bytes().decode('utf-8')
    return reviewer.original_prompt(gate, *args, **kwargs)


_prompt = handoff.hardware_prompt
hardware_prompt = FunctionType(
    _prompt.__code__, dict(_prompt.__globals__, scoped_prompt=utf8_scoped_prompt),
    _prompt.__name__, _prompt.__defaults__, _prompt.__closure__,
)
hardware_prompt.__kwdefaults__ = _prompt.__kwdefaults__


def make_backend(name, output, *, timeout_s=180., **test_dependencies):
    backend = handoff.make_backend(name, output, timeout_s=timeout_s, **test_dependencies)
    if backend.prompt_provider is not handoff.hardware_prompt:
        raise ValueError('Frozen hardware prompt provider changed; explicit port review required')
    backend.prompt_provider = hardware_prompt
    from .upstream_refusal import install_backend
    return install_backend(backend)
