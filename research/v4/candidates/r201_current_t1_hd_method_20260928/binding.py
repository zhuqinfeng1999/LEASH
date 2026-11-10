"""Existing current T1 main with named native1080 observer/qualified writer."""
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import numpy as np
import imageio.v2 as imageio

from research.v4.candidates.r201_current_t1_method_binding_20260928 import binding as real
from research.v4.candidates.r201_current_t1_packet_native_20260928 import run_one as actual_packet
from research.v4.candidates.r2_02_real_method_hd_20260927 import binding as qualified_writer
from research.v4.candidates.panda_glyph_native_pair_20260927 import native_pair as packet
from research.v4.candidates.panda_local_cutout_native_20260927 import native_pair as guard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
VERSION=real.VERSION+'-named-native1080-20fps-v1'


def admit():
    pins=json.loads((HERE/'source_freeze.json').read_text())
    for name,digest in pins.items():
        p=Path(name) if name.startswith('/') else ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('R201 HD source drift: '+name)
    return pins


def native_frame(scene,metadata):
    """Same named-camera call as the actual673 packet proof, no refresh."""
    b=scene.backend
    base=guard.immutable_snapshot(b)
    raw=scene.raw
    buffers={key:np.asarray(value).copy() for key,value in raw.items() if isinstance(value,np.ndarray)}
    checks=[]
    with guard.preserved_camera(b,checks,'R201_method_named_native1080'):
        frame=np.ascontiguousarray(b.sim.render(width=1920,height=1080,camera_name=scene.spectator_view)[::-1])
    if frame.shape!=(1080,1920,3) or frame.dtype!=np.uint8:
        raise RuntimeError('R201 method requires native1080, no square/upscale fallback')
    if scene.raw is not raw or any(not packet.exact(value,np.asarray(raw[key])) for key,value in buffers.items()):
        raise RuntimeError('Offline render mutated cached public/proprio')
    guard.same_physics(base,guard.immutable_snapshot(b))
    metadata['native_getter_calls']+=1
    metadata['all_render_guards_passed']=all(x['context_restored_exact'] and x['integration_exact'] for x in checks)
    return frame


def wrap_scene(factory,metadata):
    def selected(seed=0,*,camera_observations=True,render_gpu_device_id=0):
        scene=factory(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        def spectator(current):
            return native_frame(current,metadata)
        scene.spectator_rgb=MethodType(spectator,scene)
        return scene
    return selected


def build_entry(*,version=VERSION):
    if version!=VERSION:
        raise ValueError('Unknown R201 HD method version')
    admit()
    base=real.build_entry()
    metadata=dict(instance='R2-01',version=VERSION,width=1920,height=1080,fps=20,
                  native_getter_calls=0,encoded_frames=0,offline_only=True,
                  public_protocol='two640x480 RGB-D/K/T/proprio/common latch',
                  cached_public_not_forced_recapture=True,all_render_guards_passed=True,
                  writer_configuration=None,natural_method_ready=False,launch_authorized=False)
    text=base.selection_text
    old="    report=dict(schema_version='v4.2-dev',protocol_revision=PROTOCOL"
    if text.count(old)!=1:
        raise RuntimeError('Original actual report metadata seam changed')
    text=text.replace(old,"    report=dict(method_video_configuration=METHOD_VIDEO_METADATA,schema_version='v4.2-dev',protocol_revision=PROTOCOL",1)
    ns=dict(base.__globals__,Scene=wrap_scene(base.__globals__['Scene'],metadata),
            hashes=admit,PROTOCOL=VERSION,METHOD_VIDEO_METADATA=metadata)
    exec(compile(text,str(HERE/'binding.py')+'::original-T1-HD-metadata','exec'),ns)
    stack=ExitStack()
    stack.enter_context(patch.object(imageio,'get_writer',qualified_writer.writer_selector(imageio.get_writer,metadata)))
    ns['main'].native_method_ready=False
    ns['main'].launch_authorized=False
    return ns['main'],stack,metadata
