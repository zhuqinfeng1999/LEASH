"""Exact E2-03 thin T5 configuration: preserve proven original dog physics."""
from pathlib import Path
import xml.etree.ElementTree as ET
from research.v4.candidates.e2_native_event_20260926 import facade as base
HERE=Path(__file__).resolve().parent
class _PaperEnv(base._PetTabletopEnv):
    def _load_model(self):
        super()._load_model()
        ET.SubElement(self.model.asset,'texture',name='e203_authored_cardboard_tex',type='2d',file=str(HERE/'assets/cardboard_fibres.png'))
        matches=[]
        for material in self.model.asset.findall('material'):
            name=material.get('name','')
            if name.startswith('cardboard_box_') and ('blue_storage_poly' in name or 'inner_shadow' in name):
                material.set('texture','e203_authored_cardboard_tex');material.set('texrepeat','2 1');material.set('rgba','1 1 1 1');material.set('specular','.025');matches.append(name)
        if len(matches)!=2:raise RuntimeError('display cardboard material binding changed')
        body=self.model.worldbody.find(f".//body[@name='{self.objects_dict['cardboard_box'].root_body}']")
        if body is None:raise RuntimeError('cardboard visual parent absent')
        for i,(x,y)in enumerate(((-.268,-.167),(-.268,.167),(.268,-.167),(.268,.167))):
            ET.SubElement(body,'geom',name=f'e203_cardboard_corner_seam_{i}',type='capsule',fromto=f'{x} {y} .012 {x} {y} .156',size='.0007',rgba='.37 .26 .16 1',group='1',contype='0',conaffinity='0')
class E2CardboardDogScene(base.E2TabletopFacade):
    protocol_version='e2-cardboard-flowerpot-dog-assisted-t5-v42'
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        # Existing facade has no factory argument. Constructor-only injection in
        # isolated worker processes avoids duplicating its execution/event code.
        original=base._PetTabletopEnv;base._PetTabletopEnv=_PaperEnv
        try:super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        finally:base._PetTabletopEnv=original
        m=self.backend.sim.model;sid=m.site_name2id('gripper0_right_grip_site_cylinder');m.site_rgba[sid,3]=0
        if camera_observations:self.raw=self.backend._get_observations(force_update=True)
