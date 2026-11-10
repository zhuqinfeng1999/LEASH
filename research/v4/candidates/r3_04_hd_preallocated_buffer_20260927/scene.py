"""Offline HD only: allocate the unchanged R3 display buffer before GL startup."""
import xml.etree.ElementTree as ET
from unittest.mock import patch
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as carrier
from research.v4.candidates.r3_04_foil_sandwich_20260926.scene import T1R3FoilSandwichScene as OriginalScene
class _HDFixtureEnv(carrier._R3FixtureEnv):
    def _load_model(self):
        super()._load_model()
        visual=self.model.root.find('visual')
        if visual is None:visual=ET.SubElement(self.model.root,'visual')
        settings=visual.find('global')
        if settings is None:settings=ET.SubElement(visual,'global')
        settings.set('offwidth','1920');settings.set('offheight','1080')
class DisplayScene(OriginalScene):
    offline_buffer_protocol='initial-xml-offscreen1920x1080-v1'
    def __init__(self,*args,**kwargs):
        with patch.object(carrier,'_R3FixtureEnv',_HDFixtureEnv):
            super().__init__(*args,**kwargs)
