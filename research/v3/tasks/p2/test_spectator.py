"""P2 spectator framing is isolated from the two frozen public cameras."""
import unittest
import xml.etree.ElementTree as ET

from .scene import configure_spectator


class P2SpectatorTests(unittest.TestCase):
    def test_only_spectator_camera_changes(self):
        world = ET.fromstring('''<worldbody>
          <camera name="agentview" pos="1 2 3" quat="1 0 0 0" fovy="45"/>
          <camera name="robot0_eye_in_hand" pos="4 5 6" quat="1 0 0 0" fovy="45"/>
          <camera name="research_overview" pos="7 8 9" fovy="52"/>
          <camera name="research_top" pos="0 0 2.15" quat="1 0 0 0" fovy="52"/>
        </worldbody>''')
        before = {name: ET.tostring(world.find(f"camera[@name='{name}']"))
                  for name in ('agentview', 'robot0_eye_in_hand')}
        configure_spectator(world)
        cameras = {item.get('name'): item for item in world.findall('camera')}
        self.assertEqual(set(cameras), {'agentview','robot0_eye_in_hand','spectator'})
        for name in before:
            self.assertEqual(ET.tostring(cameras[name]), before[name])
        self.assertEqual(cameras['spectator'].get('pos'), '0.5 0 2.3')
        self.assertEqual(cameras['spectator'].get('quat'), '1 0 0 0')
        self.assertEqual(cameras['spectator'].get('fovy'), '55')


if __name__ == '__main__':
    unittest.main()
