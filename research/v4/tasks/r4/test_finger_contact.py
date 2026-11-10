"""Regression for compiled Panda finger names in R4/R5 private evaluators."""
from types import SimpleNamespace
import unittest

from research.v4.tasks.r4 import evaluator as r4
from research.v4.tasks.r5 import evaluator as r5


GEOMS = {
    "left_main": "gripper0_right_finger1_collision",
    "left_pad": "gripper0_right_finger1_pad_collision",
    "right_main": "gripper0_right_finger2_collision",
    "right_pad": "gripper0_right_finger2_pad_collision",
}


def held(module, finger_keys, bottom=.945, distance=0., reversed_pair=False):
    names = ["item_collision", *(GEOMS[key] for key in finger_keys)]
    contacts = [SimpleNamespace(
        geom1=i if reversed_pair else 0,
        geom2=0 if reversed_pair else i,
        dist=distance) for i in range(1, len(names))]
    sim = SimpleNamespace(
        model=SimpleNamespace(geom_id2name=lambda gid: names[gid]),
        data=SimpleNamespace(
            contact=contacts,
            ncon=len(contacts),
            get_joint_qpos=lambda _joint: [0., 0., bottom, 1., 0., 0., 0.]))
    obj = SimpleNamespace(contact_geoms=["item_collision"], joints=["item_joint"])
    return module._held(sim, obj, 0.)


class PandaFingerContactTests(unittest.TestCase):
    def test_distinct_fingers_and_strict_lift(self):
        for module in (r4, r5):
            with self.subTest(task=module.__name__):
                self.assertEqual(module._finger_side(GEOMS["left_main"]), "l")
                self.assertEqual(module._finger_side(GEOMS["left_pad"]), "l")
                self.assertEqual(module._finger_side(GEOMS["right_main"]), "r")
                self.assertEqual(module._finger_side(GEOMS["right_pad"]), "r")
                self.assertIsNone(module._finger_side("some_other_right_finger_collision"))
                for key in GEOMS:
                    self.assertFalse(held(module, [key]), key)
                self.assertTrue(held(module, ["left_pad", "right_pad"]))
                self.assertTrue(held(module, ["left_pad", "right_pad"], reversed_pair=True))
                self.assertTrue(held(module, ["left_main", "right_main"]))
                self.assertFalse(held(module, ["left_pad", "right_pad"], bottom=.920))
                self.assertTrue(held(module, ["left_pad", "right_pad"], bottom=.9201))
                self.assertTrue(held(module, ["left_pad", "right_pad"], distance=.002))
                self.assertFalse(held(module, ["left_pad", "right_pad"], distance=.0021))


if __name__ == "__main__":
    unittest.main()
