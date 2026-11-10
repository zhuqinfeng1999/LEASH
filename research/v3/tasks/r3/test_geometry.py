"""CPU regression for fitting the official metal pot through the microwave cavity."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[4]
POT_XML = ROOT / "data/libero/assets/turbosquid_objects/moka_pot/moka_pot.xml"


def contact_box_corners():
    tree = ET.parse(POT_XML)
    corners = []
    for geom in tree.findall(".//body[@name='object']/geom"):
        if geom.get("type") != "box" or geom.get("group") != "0":
            continue
        half = np.fromstring(geom.attrib["size"], sep=" ")
        center = np.fromstring(geom.attrib["pos"], sep=" ")
        quat = np.fromstring(geom.attrib.get("quat", "1 0 0 0"), sep=" ")
        local_rot = Rotation.from_quat(quat[[1, 2, 3, 0]])
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    corners.append(center + local_rot.apply(half * (x, y, z)))
    if not corners:
        raise RuntimeError("Official pot has no contact boxes")
    return np.asarray(corners)


class MicrowaveFitTest(unittest.TestCase):
    def test_tilted_pot_fits_cavity_with_script_waypoint(self):
        corners = contact_box_corners()
        # Heating site is (0, .01, .096) in the microwave body frame.
        # The scripted final center subtracts 4 cm in x and 2.4 cm in y.
        center = np.array([-.04, .01 - .024, .096])
        lower = np.array([-.1435, -.0865, .024])
        upper = np.array([.0635, .0805, .168])
        for angle in (-60, -58, 58, 60):
            with self.subTest(angle_deg=angle):
                fitted = Rotation.from_euler("y", angle, degrees=True).apply(corners) + center
                self.assertTrue(np.all(fitted.min(axis=0) > lower), fitted.min(axis=0))
                self.assertTrue(np.all(fitted.max(axis=0) < upper), fitted.max(axis=0))
        self.assertGreater(np.ptp(corners[:, 2]), upper[2] - lower[2])


if __name__ == "__main__":
    unittest.main()
