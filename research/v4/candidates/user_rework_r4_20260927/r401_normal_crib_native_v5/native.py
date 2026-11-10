"""Own149 bag proof; four actual stages, legacy named offline camera only."""
import ast,hashlib,inspect,json
from pathlib import Path
from types import FunctionType,SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET
from research.v4.candidates.r4_01_eligible_pickup_native_20260927 import native as template
from ..r401_range_clearance_slot_v4 import scene,scripts,qualify
HERE=Path(__file__).resolve().parent;ROOT=scene.ROOT
FIT=json.loads((HERE/'camera_fit.json').read_text())
def offline_camera(world):
 if not FIT['frustum_bounding_pass']or FIT['camera_name']!='v4_r4_spectator':raise RuntimeError('Own actual offline camera fit required')
 if world.find("camera[@name='v4_r4_spectator']")is not None:raise RuntimeError('Duplicate legacy spectator')
 ET.SubElement(world,'camera',name=FIT['camera_name'],pos=' '.join(map(str,FIT['position'])),xyaxes=' '.join(map(str,FIT['xyaxes'])),fovy=str(FIT['fovy_deg']))
INIT_NS={**scene.SELECTED_INIT.__globals__,'_camera':offline_camera}
NATIVE_INIT=FunctionType(scene.SELECTED_INIT.__code__,INIT_NS,argdefs=scene.SELECTED_INIT.__defaults__)
class NativeScene(scene.Scene):
 def __init__(self,*a,**kw):
  with patch.object(scene,'SELECTED_INIT',NATIVE_INIT):super().__init__(*a,**kw)
def own_run_case(seed,case):
 with patch.object(scripts,'Scene',SELECTED_FACTORY):return scripts.run_case(seed,case)
SELECTED_FACTORY=NativeScene
class Configuration:
 def __init__(self,instance):
  if instance!='R4-01':raise ValueError('Exact own bag only')
  self.Scene=NativeScene;self.namespace=globals();self.scene_token='SELECTED_FACTORY';self.run_case=own_run_case
def check_qualification():
 q=json.loads((scripts.HERE/'qualification30_summary.json').read_text())
 if q['source_sha256']!=qualify.hashes()or q['script_cases_passed']!=30 or q['initial_unique_seeds']!=10:raise RuntimeError('Exact own new30scripts required')
 return q
def hashes():
 pins={**template.hashes(),**qualify.hashes()}
 for p in [*HERE.glob('*.py'),HERE/'camera_fit.json',HERE/'cpu_contract.json',scripts.HERE/'qualification30_summary.json',ROOT/'envs/libero/lib/python3.11/site-packages/robosuite/utils/camera_utils.py',ROOT/'envs/libero/lib/python3.11/site-packages/robosuite/utils/binding_utils.py']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
 if f.exists()and pins!=json.loads(f.read_text()):raise RuntimeError('Own native source drift')
 return pins
source=template.source
old="kwargs.update(camera_observations=True,render_gpu_device_id=a.egl_device);super().__init__(*args,**kwargs);report['physical_gpu']=observe_physical_gpu(os.getpid(),a.physical_gpu)\n    if not report['physical_gpu']['observed']:factory.close(self);raise RuntimeError('Actual physical GPU/PID unverified')\n    try:self._initialize_stage()\n    except BaseException:factory.close(self);raise"
new="kwargs.update(camera_observations=True,render_gpu_device_id=a.egl_device);super().__init__(*args,**kwargs)\n    try:\n     self._initialize_stage();report['physical_gpu']=observe_physical_gpu(os.getpid(),a.physical_gpu)\n     if not report['physical_gpu']['observed']:raise RuntimeError('Actual physical GPU/PID unverified')\n    except BaseException:factory.close(self);raise"
if source.count(old)!=1:raise RuntimeError('Original actual-render-before-PID seam changed')
source=source.replace(old,new)
before='before=rec._state(self);packet=self.capture_agent();meta={};sim=self.backend.sim'
after="sim=self.backend.sim;camera_arrays={n:np.asarray(getattr(sim.model._model,n)).copy()for n in('cam_pos','cam_quat','cam_fovy')};before=rec._state(self);packet=self.capture_agent();meta={}"
if source.count(before)!=1:raise RuntimeError('Original capture guard seam changed')
source=source.replace(before,after)
before="if not equal:raise RuntimeError('Stage capture changed integration')"
after="camera_equal=all(np.array_equal(v,np.asarray(getattr(sim.model._model,n)))for n,v in camera_arrays.items())\n    if not equal or not camera_equal:raise RuntimeError('Stage capture changed integration/camera model')"
if source.count(before)!=1:raise RuntimeError('Original integration guard changed')
source=source.replace(before,after).replace('integration_equal=equal,spectator_resolution=[640,480]','integration_equal=equal,camera_model_equal=camera_equal,spectator_camera=FIT,spectator_resolution=[640,480]')
namespace={**template.namespace,'HERE':HERE,'ROOT':ROOT,'engine':SimpleNamespace(Configuration=Configuration),'hashes':hashes,'check_qualification':check_qualification}
namespace['FIT']=FIT
exec(compile(ast.parse(source),__file__+':own-original650-native-stage-driver','exec'),namespace)
main=namespace['main']
if __name__=='__main__':raise SystemExit(main())
