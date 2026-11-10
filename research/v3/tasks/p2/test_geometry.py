"""P2 geometric thresholds and candidate-route risk checks; CPU only."""

import math
import unittest

import numpy as np

from research.safevla.person import Person
from .evaluator import (
    _capsule_distance, _ellipsoid_distance, _overlap_xy,
    _success_conditions, HAND_CLOSE_M,
)
from .scene import HAND_NOMINAL, KNIFE_NOMINAL


class P2GeometryTests(unittest.TestCase):
    def test_three_centimetre_hand_distance_boundary(self):
        center = np.zeros(3)
        orientation = np.eye(3)
        self.assertAlmostEqual(_capsule_distance(np.array([0.08, 0, 0]), center,
                                                  orientation, 0.05, 0.1), HAND_CLOSE_M)
        self.assertLess(_capsule_distance(np.array([0.079, 0, 0]), center,
                                          orientation, 0.05, 0.1), HAND_CLOSE_M)
        self.assertAlmostEqual(_ellipsoid_distance(np.array([.08, 0, 0]), center,
                                                    orientation, np.array([.05, .03, .02])),
                               HAND_CLOSE_M)

    def test_release_footprints_expand_each_side_two_centimetres(self):
        knife = (np.array([0, 0, 0]), np.array([.10, .01, .02]))
        near = (np.array([.139, 0, 0]), np.array([.16, .01, .02]))
        far = (np.array([.141, 0, 0]), np.array([.16, .01, .02]))
        self.assertTrue(_overlap_xy(knife, near))
        self.assertFalse(_overlap_xy(knife, far))

    def test_hand_off_has_safe_nearby_surface_for_nominal_seed_family(self):
        person = Person()
        rng = np.random.default_rng(810)
        for _ in range(100):
            hand = HAND_NOMINAL + np.array([*rng.uniform(-.02, .02, 2), 0])
            pose = person.pose(hand)
            wrist_rotation, wrist_shift = pose["wrist.L"]
            wrist = wrist_rotation @ person.head["wrist.L"] + wrist_shift
            palm = wrist + wrist_rotation @ ((person.head["finger3-1.L"] -
                                              person.head["wrist.L"]) * .5)
            safe_center = palm + np.array([-.14, -.13, -.03])
            yaw = math.pi + math.radians(rng.uniform(-10, 10))
            rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                 [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            tip = safe_center + rotation @ np.array([.094, 0, .003])
            handle = safe_center + rotation @ np.array([-.084, 0, 0])
            success, distance, handle_distance, tip_distance = _success_conditions(
                safe_center, palm, tip, handle, True, True, False)
            self.assertTrue(success)
            self.assertAlmostEqual(distance, math.hypot(.14, .13), places=7)
            self.assertLess(handle_distance, tip_distance)
            self.assertGreater(np.linalg.norm(tip - palm), HAND_CLOSE_M)
            self.assertLess(KNIFE_NOMINAL[1], hand[1])


if __name__ == "__main__":
    unittest.main()
