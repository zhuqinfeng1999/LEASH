"""Receiving-machine gate protocol; experiment scores never set thresholds."""
from research.v4.development.confidence_gate_dev_20261006.core import Thresholds

CURRENT_THRESHOLDS = {'G1': .5, 'G2': .5, 'G3': .3}
PROTOCOL_REVISION = 'operator-selected-g1-50-g2-50-g3-30-20261022'


def configured_thresholds(config):
    values = config.get('thresholds', CURRENT_THRESHOLDS)
    if not isinstance(values, dict) or set(values) != {'G1', 'G2', 'G3'}:
        raise ValueError('Explicit G1/G2/G3 threshold mapping required')
    return Thresholds(values['G1'], values['G2'], values['G3'])
