"""Frozen v3 geometry thresholds; no simulator launch."""

import numpy as np

from .evaluator import _held_from_contacts, _overlap_xy


def test_footprints_expand_two_centimetres_each():
    a = (np.array([0.00, 0.00, 0]), np.array([0.10, 0.10, 1]))
    close = (np.array([0.139, 0.05, 0]), np.array([0.20, 0.10, 1]))
    far = (np.array([0.141, 0.05, 0]), np.array([0.20, 0.10, 1]))
    assert _overlap_xy(a, close)
    assert not _overlap_xy(a, far)


def test_hold_requires_two_finger_contacts_and_more_than_two_centimetres():
    table = 0.91
    assert not _held_from_contacts({1, 2}, table + 0.02, table)
    assert _held_from_contacts({1, 2}, table + 0.02001, table)
    assert not _held_from_contacts({1}, table + 0.03, table)
