"""Small offline spectator recorder; never an online camera or controller."""
import hashlib,subprocess,time
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
FONT=Path('/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf')
FONT_NOTICE=Path('/usr/share/doc/fonts-noto-core/copyright')
class NativeSceneRecorder:
    def __init__(self,path,*,fps=20,width=1920,height=1080,stride=2):
        self.path=Path(path);self.fps=fps;self.width=width;self.height=height;self.stride=stride;self.frame_count=0;self.captures=0;self.render_seconds=0.;self.encode_write_seconds=0.;self.segment=[];self.render_state_checks=[];self.resolution_attempts=[];self.pipe=None;self.font=None
    def _state(self,scene):
        m,d=scene.backend.sim.model._model,scene.backend.sim.data._data;sig=mujoco.mjtState.mjSTATE_INTEGRATION;out=np.zeros(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,out,sig);return out
    def native_frame(self,scene):
        before=self._state(scene);started=time.monotonic()
        try:frame=scene.backend.sim.render(width=self.width,height=self.height,camera_name=scene.spectator_view)[::-1]
        except BaseException as exc:
            self.resolution_attempts.append(dict(width=self.width,height=self.height,error=f'{type(exc).__name__}: {exc}'))
            if self.pipe is not None or(self.width,self.height)!=(1920,1080):raise
            self.width,self.height=1280,720
            frame=scene.backend.sim.render(width=self.width,height=self.height,camera_name=scene.spectator_view)[::-1]
        self.render_seconds+=time.monotonic()-started;self.captures+=1
        if frame.shape!=(self.height,self.width,3)or frame.dtype!=np.uint8:raise ValueError('native renderer must return requested resolution; no upscaling')
        same=np.array_equal(before,self._state(scene))
        if not same:raise RuntimeError('spectator render changed integration state')
        if self.captures<=2 or self.captures%100==0:self.render_state_checks.append(dict(capture=self.captures,integration_state_equal=same))
        return np.ascontiguousarray(frame)
    def _ensure_encoder(self):
        if self.pipe is not None:return
        self.font=ImageFont.truetype(str(FONT),max(24,int(self.height*.032)))
        self.small=ImageFont.truetype(str(FONT),max(19,int(self.height*.026)))
        self.stderr=self.path.with_suffix('.ffmpeg.log').open('w')
        cmd=['ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{self.width}x{self.height}','-r',str(self.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)]
        self.pipe=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.stderr)
    def write(self,frame,*,instruction,chapter,line,repeat=1):
        self._ensure_encoder();im=Image.fromarray(frame).convert('RGBA');overlay=Image.new('RGBA',im.size);d=ImageDraw.Draw(overlay);pad=int(self.height*.025);bottom_h=int(self.height*.075);top_h=int(self.height*.11)
        d.rectangle((0,0,self.width,top_h),fill=(8,14,22,216));d.text((pad,pad//2),instruction,font=self.font,fill=(255,255,255,255));d.text((pad,int(self.height*.056)),chapter,font=self.small,fill=(166,211,255,255))
        d.rectangle((0,self.height-bottom_h,self.width,self.height),fill=(8,14,22,216));d.text((pad,self.height-bottom_h+pad//2),line,font=self.small,fill=(255,241,200,255));image=Image.alpha_composite(im,overlay).convert('RGB');raw=image.tobytes();started=time.monotonic()
        for _ in range(repeat):self.pipe.stdin.write(raw);self.frame_count+=1
        self.encode_write_seconds+=time.monotonic()-started
    def begin_segment(self,scene,*,instruction,chapter):
        packet=scene.capture_agent();cameras={n:dict(rgb_shape=list(c['rgb'].shape),depth_shape=list(c['depth_m'].shape),intrinsics_sha256=hashlib.sha256(np.asarray(c['intrinsics']).tobytes()).hexdigest(),extrinsics_sha256=hashlib.sha256(np.asarray(c['extrinsics']).tobytes()).hexdigest())for n,c in packet['cameras'].items()}
        if set(cameras)!={'agentview','robot0_eye_in_hand'}or any(v['rgb_shape']!=[480,640,3]or v['depth_shape']!=[480,640]for v in cameras.values()):raise ValueError('original two640x480RGBD protocol changed')
        frame=self.native_frame(scene);self.write(frame,instruction=instruction,chapter=chapter,line='Scripted scene inspection | No GPT / VLA model calls | 2x replay',repeat=self.fps)
        record=dict(chapter=chapter,start_frame=self.frame_count-self.fps,physical_start_time_s=float(scene.backend.sim.data.time),public_camera_metadata=cameras,controls=0);self.segment.append(record);step=scene.step
        def observed_step(action):
            result=step(action);record['controls']+=1
            if record['controls']%self.stride==0:
                frame=self.native_frame(scene);line=f"Scripted motion | simulation {scene.backend.sim.data.time:.1f}s | 2x replay | no model"
                self.write(frame,instruction=instruction,chapter=chapter,line=line)
            return result
        scene.step=observed_step;close=scene.close
        def closing():
            try:
                frame=self.native_frame(scene);self.write(frame,instruction=instruction,chapter=chapter,line='End of scripted trajectory | Outcome is listed in the scene-review JSON',repeat=self.fps)
                record.update(end_frame=self.frame_count,physical_end_time_s=float(scene.backend.sim.data.time))
            finally:close()
        scene.close=closing
    def close(self):
        if self.pipe is None:return
        self.pipe.stdin.close()
        try:code=self.pipe.wait(timeout=30)
        except subprocess.TimeoutExpired:self.pipe.kill();self.pipe.wait();raise
        finally:self.stderr.close()
        if code:raise RuntimeError(f'ffmpeg encoder failed: {code}')
