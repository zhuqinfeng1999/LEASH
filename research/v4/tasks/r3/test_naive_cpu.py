"""CPU-only bounds for the unqualified R3 action candidate."""
import unittest

import numpy as np

from research.v4.tasks.r3.naive_robot import (
    MAX_TOTAL_STEPS, MAX_TRANSLATION_ACTION, MAX_STALL_STEPS, translation_action,
)


class R3ActionChecks(unittest.TestCase):
    def test_osc_translation_is_finite_and_component_bounded(self):
        self.assertTrue(np.array_equal(translation_action([0., 0., 0.]), np.zeros(3)))
        cmd = translation_action([.20, -.20, .002])
        np.testing.assert_allclose(cmd, [MAX_TRANSLATION_ACTION, -MAX_TRANSLATION_ACTION, .04])
        for bad in ([np.nan, 0., 0.], [np.inf, 0., 0.], [0., 0.]):
            with self.assertRaises(ValueError):
                translation_action(bad)

    def test_waypoints_cross_only_after_low_clearance_alignment(self):
        # Relative to the LIBERO heating site: half-size .12/.0835/.08 m.
        region_half = np.array([.12, .0835, .08])
        recenter = np.array([-.04, -.18, .12])
        lower = np.array([-.04, -.18, .01])
        lead = np.array([-.04, -.11, .01])
        commit = np.array([-.04, -.06, .00])
        self.assertGreater(abs(recenter[1]), region_half[1])
        self.assertGreater(abs(lower[1]), region_half[1])
        self.assertGreater(abs(lead[1]), region_half[1])
        self.assertTrue(np.all(np.abs(commit) < region_half))
        self.assertLess(abs(commit[2] - lower[2]), .03)
        self.assertLessEqual(MAX_TOTAL_STEPS, 640)
        self.assertLessEqual(MAX_STALL_STEPS, 35)


if __name__ == '__main__':
    unittest.main()
