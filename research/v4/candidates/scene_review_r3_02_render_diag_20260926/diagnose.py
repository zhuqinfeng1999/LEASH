"""Zero-control RGB resize reproducibility diagnostic; no weakened guard."""
import json,os,hashlib,time,signal
from pathlib import Path
import numpy as np
from research.v4.candidates.r3_02_cc0_soup_config_20260926.scene import SoupCanScene
from research.v4.candidates.scene_review_export_20260926.recorder import NativeSceneRecorder
HERE=Path(__file__).resolve().parent

def compare(a,b):
 out={}
 for n,c in a['cameras'].items():
  out[n]={}
  for k in ('rgb','depth_m','intrinsics','extrinsics'):
   target=b['cameras'][n][k];delta=np.abs(c[k].astype(float)-target.astype(float))
   out[n][k]=dict(equal=bool(np.array_equal(c[k],target)),changed_elements=int(np.count_nonzero(c[k]!=target)),max_abs_delta=float(delta.max()))
 return out
def main():
 start=time.monotonic();signal.alarm(120);scene=SoupCanScene(seed=0,camera_observations=True,render_gpu_device_id=0);r=NativeSceneRecorder(HERE/'unused.mp4');state=r._state(scene);out={'pid':os.getpid(),'controls':0,'models':[],'source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 try:
  a=scene.capture_agent();b=scene.capture_agent();out['native640_repeat_before_resize']=compare(a,b)
  first=r.native_frame(scene);after=scene.capture_agent();out['first1080_resize']=compare(b,after)
  stable1=scene.capture_agent();second=r.native_frame(scene);stable2=scene.capture_agent();out['second1080_same_buffer']=compare(stable1,stable2)
  third=r.native_frame(scene);stable3=scene.capture_agent();out['third1080_same_buffer']=compare(stable2,stable3);out['1080_frames_equal']=bool(np.array_equal(second,third));out['integration_exact']=bool(np.array_equal(state,r._state(scene)))
  from PIL import Image
  Image.fromarray(first).save(HERE/'first_native1080.png');out['elapsed_s']=time.monotonic()-start
  (HERE/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
 finally:scene.close();signal.alarm(0)
if __name__=='__main__':main()
