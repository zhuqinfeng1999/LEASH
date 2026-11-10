"""Same natural S101 production factory, scoped XML material substitution."""
import hashlib,json
from unittest.mock import patch
import mujoco
from robosuite.utils.binding_utils import MjSim
from research.v4.candidates.s1_v42_tabletop_20260925.scene_v5_stable_pan300 import S1StablePan300Facade
from .helper import HERE,ROOT,patched_xml
ORIGINAL=MjSim.from_xml_string
def initialized(which,camera_observations=False,egl=-1):
 if which not in('before','after'):raise ValueError(which)
 calls=[]
 def factory(cls,xml):
  chosen=xml
  if which=='after':chosen,_=patched_xml(xml)
  calls.append({'incoming_xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),'variant':which})
  return ORIGINAL(chosen)
 original_descriptor=MjSim.__dict__['from_xml_string']
 with patch.object(MjSim,'from_xml_string',classmethod(factory)):
  s=S1StablePan300Facade(seed=0,camera_observations=camera_observations,render_gpu_device_id=egl)
 if len(calls)!=1 or MjSim.__dict__['from_xml_string']is not original_descriptor:s.close();raise RuntimeError('Scoped production factory mismatch')
 s._garment_compile_calls=calls
 return s
def hashes():
 files=[*HERE.glob('*.py'),HERE/'geometry_contract.json',HERE/'material_partition.json',HERE/'NOTICE.md',HERE/'HANDOFF.md',
  ROOT/'research/safevla/person.py',ROOT/'research/safevla/character_assets.py',
  ROOT/'research/upstream/makehuman-character/selected-manifest.json',ROOT/'research/upstream/makehuman-character/casual06-provenance.json',
  ROOT/'research/release/LICENSES/CC0-1.0.txt',ROOT/'research/v4/candidates/garment_unbranded_source_readonly_20260927/HANDOFF.md',
  ROOT/'research/v4/candidates/garment_unbranded_source_readonly_20260927/analysis.json']
 from .helper import SOURCE,SOURCE_HASHES
 files += [SOURCE/k for k in SOURCE_HASHES]
 selected=json.loads((ROOT/'research/upstream/makehuman-character/selected-manifest.json').read_text())
 files += [ROOT/r['path']for r in selected['files']]
 files += [ROOT/'research/upstream/makehuman-assets'/name for name in ('base.obj','base.mhclo','default.mhmat','default.mhskel','default_weights.mhw','LICENSE.ASSETS.md','LICENSE.md','commit.txt')]
 files += [ROOT/'research/v4/candidates/panda_glyph_native_pair_20260927/native_pair.py',
  ROOT/'research/v4/candidates/panda_local_cutout_native_20260927/native_pair.py',
  ROOT/'research/v4/candidates/panda_local_cutout_native_20260927/proof.py',
  ROOT/'envs/libero/lib/python3.11/site-packages/robosuite/utils/binding_utils.py',
  ROOT/'envs/libero/lib/python3.11/site-packages/robosuite/utils/camera_utils.py',HERE/'cpu_name_offset_guard_failure.json',HERE/'dispatch_contract.json']
 from research.v4.candidates.s1_v42_engineering_qualification_20260926 import qualify_current
 pins=dict(qualify_current.hashes())
 for p in files:pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists()and json.loads(f.read_text())!=pins:raise RuntimeError('Own garment source drift')
 return pins
