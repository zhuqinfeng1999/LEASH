"""Offline spectator getter/writer opt-in around the original REAL R202 main."""
import hashlib,json
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType,MethodType
from unittest.mock import patch
import numpy as np
import imageio.v2 as imageio
from research.v4.candidates.release_r202_method_optin_20260927 import binding as real
from research.v4.candidates.release_r202_method_optin_20260927 import freeze as real_sources
from research.v4.candidates.scene_review_r2_variants_20260927 import export as hd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
VERSION='R202-real-method-offline-native1920x1080-20fps-optin-v1'
VIDEO_RECEIPT=ROOT/'research/v4/review_videos/R2-02.json'
PUBLIC_KEYS=('agentview_image','agentview_depth','robot0_eye_in_hand_image',
    'robot0_eye_in_hand_depth','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def source_union():
    pins=dict(real_sources.admit_sources())
    for name,digest in hd.hashes('R2-02').items():
        if name in pins and pins[name]!=digest:raise RuntimeError('R202 HD source conflict: '+name)
        if sha(ROOT/name)!=digest:raise RuntimeError('R202 HD dependency drift: '+name)
        pins[name]=digest
    for path in [VIDEO_RECEIPT,*HERE.glob('*.py'),*HERE.glob('*.md')]:
        pins[str(path.relative_to(ROOT))]=sha(path)
    return dict(sorted(pins.items()))
def source_hashes():
    actual=source_union();path=HERE/'source_freeze.json'
    if not path.is_file() or actual!=json.loads(path.read_text()):raise RuntimeError('R202 HD whole-source admission failed')
    return actual

class Native1080Observer(hd.guard.LateralRecorder):
    """Existing qualified getter only; never begin_segment or sensor refresh."""
    def native_frame(self,scene):
        raw=scene.raw
        buffers={key:np.asarray(raw[key]).copy() for key in PUBLIC_KEYS if key in raw}
        frame=super().native_frame(scene)
        if (self.width,self.height)!=(1920,1080) or frame.shape!=(1080,1920,3) or frame.dtype!=np.uint8:
            raise ValueError('R202 method requires genuinely rendered native1080; fallback/upscale forbidden')
        if scene.raw is not raw or any(not np.array_equal(value,np.asarray(raw[key])) for key,value in buffers.items()):
            raise RuntimeError('Offline spectator changed current cached public/proprio buffers')
        return frame

def wrap_scene(factory,metadata):
    def selected(seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        scene=factory(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        observer=Native1080Observer(Path('spectator.mp4'),width=1920,height=1080,fps=20)
        # No GL/context/capture/step in this wrapper. The existing method calls
        # the getter at original save/sink boundaries on the owner thread.
        def spectator(current):
            frame=observer.native_frame(current);metadata['native_getter_calls']+=1;return frame
        scene.spectator_rgb=MethodType(spectator,scene)
        scene._method_hd_observer=observer
        return scene
    return selected

class NativeWriter:
    def __init__(self,writer,metadata):self.writer=writer;self.metadata=metadata
    def append_data(self,frame,*args,**kwargs):
        if not isinstance(frame,np.ndarray) or frame.shape!=(1080,1920,3) or frame.dtype!=np.uint8:
            raise ValueError('Reject nonnative method spectator frame before encoder')
        result=self.writer.append_data(frame,*args,**kwargs);self.metadata['encoded_frames']+=1;return result
    def close(self):return self.writer.close()

def writer_selector(original,metadata):
    def get_writer(path,*args,**kwargs):
        if Path(path).name!='spectator.mp4':return original(path,*args,**kwargs)
        if kwargs.get('fps')!=20:raise ValueError('R202 original spectator FPS changed')
        # Original macro_block_size=16 silently rescales1080 to1088. Only this
        # offline sink opts out; no pixels are enlarged or input protocol edited.
        selected={**kwargs,'macro_block_size':1}
        metadata['writer_configuration']=dict(fps=20,macro_block_size=1,width=1920,height=1080,
            original_sink_fps_unchanged=True,no_imageio_macroblock_upscale=True)
        return NativeWriter(original(path,*args,**selected),metadata)
    return get_writer

def build_entry(*,version=VERSION):
    if version!=VERSION:raise ValueError('Unknown R202 method HD opt-in version')
    source_hashes();base=real.runner(appearance_version=real.VERSION)
    metadata=dict(instance='R2-02',protocol=VERSION,width=1920,height=1080,fps=20,
        native_getter_calls=0,encoded_frames=0,offline_only=True,public_protocol='two640x480RGB-D',
        per_render_integration_public_model_raw_guard=True,new_forced_sensor_refresh=False,
        actual_1080_native_preview_pending=True,whole_trajectory_RGB_bit_identity_proven=False,
        public_freeze=False,launch_authorized=False,natural_method_ready=False)
    namespace=dict(base.__globals__,Scene=wrap_scene(base.__globals__['Scene'],metadata),
        hashes=source_hashes,PROTOCOL=base.__globals__['PROTOCOL']+'-'+VERSION)
    selected=FunctionType(base.__code__,namespace,base.__name__,base.__defaults__,base.__closure__)
    stack=ExitStack()
    # Import inside original main resolves this exact module. Keep this scope
    # only for the root-owned call, then close; default installed writer unchanged.
    stack.enter_context(patch.object(imageio,'get_writer',writer_selector(imageio.get_writer,metadata)))
    return selected,stack,metadata
