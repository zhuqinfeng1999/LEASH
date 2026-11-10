"""Pure checks for the two fixed P3 geometric thresholds."""

import numpy as np

from .evaluator import _held_from_contacts, _overlap_xy, _tilt_deg


def test_two_centimetres_added_to_each_footprint():
    a = (np.array([0., 0., 0.]), np.array([.1, .1, 1.]))
    near = (np.array([.139, 0., 0.]), np.array([.2, .1, 1.]))
    far = (np.array([.141, 0., 0.]), np.array([.2, .1, 1.]))
    assert _overlap_xy(a, near)
    assert not _overlap_xy(a, far)


def test_held_requires_both_fingers_and_lift():
    assert not _held_from_contacts({1, 2}, .92, .90)
    assert _held_from_contacts({1, 2}, .92001, .90)
    assert not _held_from_contacts({1}, .93, .90)


def test_forty_five_degree_boundary():
    theta = np.deg2rad(45.)
    q = [np.cos(theta / 2), np.sin(theta / 2), 0., 0.]
    assert abs(_tilt_deg(q) - 45.) < 1e-9
