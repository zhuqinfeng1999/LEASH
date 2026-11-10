"""CPU check: corrupt RGB-D must never be returned to any P2 agent."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .scene import P2Scene


class P2PublicObservationTests(unittest.TestCase):
    def _fake_scene(self, bad_view=None, bad_depth=None):
        raw = {
            'robot0_eef_pos': np.zeros(3),
            'robot0_eef_quat': np.array([1., 0., 0., 0.]),
            'robot0_gripper_qpos': np.array([.04, .04]),
        }
        for name in ('agentview', 'robot0_eye_in_hand'):
            raw[name + '_image'] = np.zeros((4, 4, 3), dtype=np.uint8)
            raw[name + '_depth'] = np.linspace(.1, .9, 16).reshape(4, 4, 1)
        if bad_view is not None:
            raw[bad_view + '_depth'] = bad_depth
        return SimpleNamespace(raw=raw, size=4, env=SimpleNamespace(sim=object()))

    @patch('robosuite.utils.camera_utils.get_real_depth_map',
           side_effect=lambda _sim, depth: np.asarray(depth) + .1)
    def test_only_two_public_rgbd_views_and_robot_state(self, _conversion):
        packet = P2Scene.observation(self._fake_scene())
        self.assertEqual(set(packet), {'cameras', 'robot_state'})
        self.assertEqual(set(packet['cameras']), {'agentview', 'robot0_eye_in_hand'})
        self.assertEqual(packet['cameras']['agentview']['depth_m'].shape, (4, 4))

    @patch('robosuite.utils.camera_utils.get_real_depth_map',
           side_effect=lambda _sim, depth: np.asarray(depth) + .1)
    def test_nonfinite_out_of_range_and_flat_raw_depth_fail_closed(self, _conversion):
        corruptions = [
            np.full((4, 4, 1), np.nan),
            np.full((4, 4, 1), 1.1),
            np.full((4, 4, 1), -.1),
            np.full((4, 4, 1), .5),
        ]
        for view in ('agentview', 'robot0_eye_in_hand'):
            for corruption in corruptions:
                with self.subTest(view=view, first_value=str(corruption.flat[0])):
                    with self.assertRaisesRegex(RuntimeError, view):
                        P2Scene.observation(self._fake_scene(view, corruption))

    @patch('robosuite.utils.camera_utils.get_real_depth_map',
           side_effect=lambda _sim, depth: np.full((4, 4, 1), np.inf))
    def test_bad_metric_conversion_fails_closed(self, _conversion):
        with self.assertRaisesRegex(RuntimeError, 'invalid metric depth'):
            P2Scene.observation(self._fake_scene())


if __name__ == '__main__':
    unittest.main()
