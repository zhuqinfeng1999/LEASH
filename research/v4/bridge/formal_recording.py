"""Separate human-view videos with an uncut, uniformly accelerated wall clock.

The observation hook does not step physics or add method inputs. Source frames
and their monotonic timestamps are retained; playback holds the last observed
state across model/reviewer waits. Encoding happens after the recording scope.
"""
from bisect import bisect_right
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess
import time

PUBLIC_VIEWS = ('agentview', 'robot0_eye_in_hand')
RECORDING = dict(directory='videos', fps=60, source_fps=20, codec='libx264', crf=18,
    public_dimensions=[640, 480], spectator_dimensions=[1920, 1080],
    views=[*PUBLIC_VIEWS, 'research_spectator'], separate_views=True,
    timeline='first real task observation through recording-scope exit, including every model/reviewer wait; uniformly accelerated with latest-observation holds',
    version='full-wall-time-cfr60-playback-v2-20261022', playback_speed=16.,
    target_playback_seconds_approximate=30.,
    duration_policy='fixed 16x for every episode; complete duration may be shorter or longer than 30 seconds',
    source_directory='keyframes', source_index='frames.jsonl', playback_directory='playback_v2',
    playback_index='playback_frames.jsonl', source_timestamps='monotonic seconds from recording-scope entry',
    unobserved_prefix='trim only before the first real observation; no black prefix and no future-state backdating',
    source_sampling='one image per distinct executed sequence; not a 60fps camera acquisition',
    presentation_sampling='60fps CFR display samples, holding the last actual observation; no generated motion or optical interpolation',
    coverage='recording-scope entry through exit; only phases enclosed by the caller, no claim about other episode phases',
    overlays=True, overlay_human_video_only=True, montage=False, upscaling=False,
    spectator_human_only=True)


def formal_output_path(root, *, experiment, condition, planner_model, method,
                       instance, seed, episode_id):
    """One fresh attempt directory; retries never overwrite an old episode."""
    parts = (experiment, condition, planner_model, method, instance, episode_id)
    if any(not isinstance(p, str) or not p or '/' in p or '\\' in p or p in {'.', '..'} for p in parts):
        raise ValueError('Output hierarchy requires ordinary single-component names')
    if type(seed) is not int:
        raise ValueError('Output hierarchy requires an explicit seed')
    return Path(root)/'research/v4/runs'/experiment/condition/planner_model/method/instance/f'seed_{seed}'/episode_id


def wall_time_samples(capture_times, elapsed_s, *, fps=60, speed=16.):
    """Uniform wall-time samples, always using a past observation or blank.

    Every episode uses the same speed. The last frame covers the remaining
    subframe interval, with at most one display frame of declared padding.
    """
    if (not math.isfinite(elapsed_s) or elapsed_s < 0 or fps <= 0
            or not math.isfinite(speed) or speed <= 0):
        raise ValueError('Invalid wall-time playback duration')
    times = list(capture_times)
    if (any(not math.isfinite(t) or t < 0 or t > elapsed_s for t in times)
            or any(b < a for a, b in zip(times, times[1:]))):
        raise ValueError('Source frame timestamps must be ordered inside the scope')
    if not times:
        return
    start_s = times[0]
    availability = [math.ceil((t-start_s)/speed*fps) for t in times]
    count = max(1, math.ceil((elapsed_s-start_s)/speed*fps))
    for i in range(count):
        wall = start_s+i*speed/fps
        yield dict(frame=i, playback_seconds=i/fps, wall_seconds=wall,
                   task_elapsed_seconds=wall-start_s,
                   wall_interval_end_seconds=min(elapsed_s, start_s+(i+1)*speed/fps),
                   source_frame=bisect_right(availability, i)-1)


@lru_cache(maxsize=4)
def _overlay_font(height):
    from PIL import ImageFont
    try:
        return ImageFont.truetype('DejaVuSans.ttf', max(12, height//40))
    except OSError:
        return ImageFont.load_default()


def _playback_overlay(frame, sample, *, elapsed_s, speed, observed_s):
    """Annotate only the human playback; source packets remain unchanged."""
    import numpy as np
    from PIL import Image, ImageDraw
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    label = (f"Wall {sample['wall_seconds']:.1f}/{elapsed_s:.1f}s | {speed:.2f}x | "
             + ('awaiting first observation' if observed_s is None
                else f'latest observation {observed_s:.1f}s; waits retained'))
    font = _overlay_font(image.height)
    draw.rectangle((0, 0, image.width, max(25, image.height//40+12)), fill=(0, 0, 0))
    draw.text((7, 6), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def verify_playback_files(destination, count):
    root = Path(__file__).resolve().parents[3]
    verified = {}
    for name in RECORDING['views']:
        path = Path(destination)/(name+'.mp4')
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,duration,avg_frame_rate,r_frame_rate',
            '-of', 'json', str(path)], text=True, capture_output=True, check=True, timeout=45)
        info = json.loads(result.stdout)['streams'][0]
        dimensions = RECORDING['spectator_dimensions'] if name == 'research_spectator' else RECORDING['public_dimensions']
        if ([info['width'], info['height']] != dimensions or info['avg_frame_rate'] != '60/1'
                or info['r_frame_rate'] != '60/1' or int(info['nb_frames']) != count
                or abs(float(info['duration'])-count/60) > .001):
            raise ValueError('Derived video failed CFR60/dimensions/frame-count/duration validation: '+name)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                h.update(block)
        verified[name] = dict(path=str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            sha256=h.hexdigest(), **info)
    return verified


def _timestamp_expression(times, start=0, end=None):
    """Balanced AVExpr lookup: source frame N -> its recorded wall timestamp."""
    end = len(times) if end is None else end
    if end-start == 1:
        return format(times[start], '.12g')
    mid = (start+end)//2
    return f'if(lt(N,{mid}),{_timestamp_expression(times,start,mid)},{_timestamp_expression(times,mid,end)})'


def _ffmpeg_playback(source_paths, times, elapsed_s, destination, count, encoder):
    """Native CPU VFR->CFR conversion; no Python image loop or invented motion."""
    speed = RECORDING['playback_speed']
    relative = [math.ceil((t-times[0])/speed*60) for t in times]
    expression = _timestamp_expression(relative)
    duration = elapsed_s-times[0]
    for name in RECORDING['views']:
        height = RECORDING['spectator_dimensions'][1] if name == 'research_spectator' else RECORDING['public_dimensions'][1]
        font_size = max(12, height//40)
        # round=up is causal: an image first becomes available at the next CFR
        # boundary, never at a display instant preceding its real timestamp.
        # tpad only holds the last actual image through the recorded final wait;
        # the explicit frame ceiling makes this a finite encoding operation.
        filters = (f"settb=expr=1/60,setpts='{expression}',"
            f'tpad=stop_mode=clone:stop=-1,fps=60:start_time=0:round=up,'
            f'drawbox=x=0:y=0:w=iw:h={max(25,font_size+12)}:color=black:t=fill,'
            f"drawtext=fontsize={font_size}:fontcolor=white:x=7:y=6:"
            f"text='Elapsed %{{eif\\:t*16\\:d}}/{duration:.1f}s | 16x | waits retained; CFR60 display'")
        script = destination/(name+'.filter.txt')
        script.write_text(filters+'\n')
        pending = destination/(name+'_pending.mp4')
        log = destination/(name+'.encode.log')
        with log.open('w') as stream:
            codec_options = (['-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(RECORDING['crf']), '-threads', '2']
                if encoder == 'libx264' else ['-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr',
                    '-cq', '18', '-b:v', '0', '-gpu', '0', '-threads', '2'])
            subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'error',
                '-threads', '1', '-i', str(source_paths[name]), '-an', '-filter_threads', '1',
                '-filter_script:v', str(script), '-frames:v', str(count), '-r', '60',
                *codec_options, '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(pending)], stdout=stream, stderr=subprocess.STDOUT, check=True,
                timeout=max(120, count/10))
        pending.replace(destination/(name+'.mp4'))


def render_wall_time_playback(source_paths, capture_times, elapsed_s, destination, *,
                             writer_factory=None, reader_factory=None, encoder='libx264'):
    """Derive CFR60/16x videos without changing any source image or timestamp.

    Used by both the live recorder's close and the offline historical repair.
    Only initialization before the first image is omitted. A CFR display is
    not a claim that the source camera sampled at 60fps.
    """
    from imageio.v2 import get_reader, get_writer
    destination = Path(destination)
    if encoder not in ('libx264', 'h264_nvenc'):
        raise ValueError('Unregistered video encoder')
    destination.mkdir(exist_ok=False)
    real_encoder = writer_factory is None or writer_factory is get_writer
    writer_factory = writer_factory or get_writer
    if reader_factory is None:
        reader_factory = lambda path: get_reader(path, input_params=['-threads', '1'])
    times = list(capture_times)
    # Validate before creating an encoder or emitting a success receipt.
    iterator = wall_time_samples(times, elapsed_s, fps=RECORDING['fps'], speed=RECORDING['playback_speed'])
    first = next(iterator, None)
    if first is None:
        return dict(playback_frames=0, playback_seconds=0., video_files=[],
                    unavailable_reason='No actual camera observation; no scene video can be produced')
    if set(source_paths) != set(RECORDING['views']):
        raise ValueError('All three original video views are required')
    start_s = times[0]
    duration_s = elapsed_s-start_s
    speed = RECORDING['playback_speed']
    count = max(1, math.ceil(duration_s/speed*RECORDING['fps']))
    def samples():
        return wall_time_samples(times, elapsed_s, fps=RECORDING['fps'], speed=speed)
    with (destination/'playback_frames.jsonl').open('w') as index:
        for sample in samples():
            index.write(json.dumps(sample)+'\n')
    if real_encoder:
        _ffmpeg_playback(source_paths, times, elapsed_s, destination, count, encoder)
    files = []
    for name in RECORDING['views']:
        if real_encoder:
            files.append(name+'.mp4')
            continue
        reader = writer = None
        pending = destination/(name+'_pending.mp4')
        previous = None
        source = None
        try:
            reader = reader_factory(str(source_paths[name]))
            writer = writer_factory(str(pending), fps=RECORDING['fps'],
                codec=RECORDING['codec'], macro_block_size=1,
                pixelformat='yuv420p', ffmpeg_params=['-crf', str(RECORDING['crf']),
                    '-preset', 'veryfast', '-threads', '2'])
            for sample in samples():
                frame_no = sample['source_frame']
                if frame_no != previous:
                    source = reader.get_data(frame_no)
                    previous = frame_no
                writer.append_data(_playback_overlay(source, sample, elapsed_s=elapsed_s,
                    speed=speed, observed_s=times[frame_no]))
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                if reader is not None:
                    reader.close()
        pending.replace(destination/(name+'.mp4'))
        files.append(name+'.mp4')
    metadata = dict(playback_frames=count, playback_seconds=count/RECORDING['fps'],
        playback_padding_seconds=count/RECORDING['fps']-duration_s/speed,
        playback_wall_start_seconds=start_s, playback_wall_end_seconds=elapsed_s,
        included_task_wall_seconds=duration_s, trimmed_initialization_seconds=start_s,
        video_files=files, initialization_trim_only=True,
        interpolation='CFR60 sampling of latest actual state across the full task wall range; no optical interpolation',
        source_frame_count_may_exceed_playback_samples=True)
    if real_encoder:
        metadata.update(container_verified=True,
            verified_views=verify_playback_files(destination, count),
            render_engine='ffmpeg timestamp AVExpr, causal ceil-to-CFR60; decoder1/encoder2',
            playback_encoder=encoder,
            encoding_quality='CRF18/veryfast' if encoder == 'libx264' else 'NVENC VBR/CQ18/p4/GPU0')
    return metadata


class SeparateViewRecorder:
    def __init__(self, output, writer_factory=None, *, reader_factory=None,
                 clock=None, utc_clock=None):
        self.clock = clock or time.monotonic
        self.utc_clock = utc_clock or (lambda: datetime.now(timezone.utc).isoformat())
        self.started = self.clock()
        self.started_utc = self.utc_clock()
        self.output = Path(output)/'videos'
        self.output.mkdir(exist_ok=False)
        (self.output/'keyframes').mkdir()
        if writer_factory is None:
            from imageio.v2 import get_writer
            writer_factory = get_writer
        if reader_factory is None:
            from imageio.v2 import get_reader
            reader_factory = lambda path: get_reader(path, input_params=['-threads', '1'])
        self.writer_factory = writer_factory
        self.reader_factory = reader_factory
        self.writers = {}
        self.frames = 0
        self.capture_times = []
        self.scenes = []
        self.last_sequence = {}
        self.index = (self.output/'frames.jsonl').open('w')
        self.closed = False

    def capture(self, adapter, packet):
        import numpy as np
        scene = adapter.scene
        if not any(scene is old for old in self.scenes):
            self.scenes.append(scene)
        segment = next(i for i, old in enumerate(self.scenes) if old is scene) + 1
        sequence = getattr(scene, 'sequence', None)
        if sequence is not None and self.last_sequence.get(segment) == sequence:
            return
        captured_s = self.clock()-self.started
        if captured_s < 0 or (self.capture_times and captured_s < self.capture_times[-1]):
            raise RuntimeError('Recording monotonic clock moved backwards')
        frames = {name: packet['cameras'][name]['rgb'] for name in PUBLIC_VIEWS}
        for name, frame in frames.items():
            if np.asarray(frame).shape != (480, 640, 3):
                raise ValueError(f'Formal video requires the existing native 640x480 {name} packet')
        # Existing MuJoCo renderer enlarges its offscreen buffer as needed. No
        # physics or public camera/observation configuration is changed here.
        # Scene families already declare their own existing spectator camera.
        # The stable output filename is independent of that camera's name.
        spectator_camera = scene.spectator_view
        frames['research_spectator'] = np.ascontiguousarray(scene.env.sim.render(
            width=1920, height=1080, camera_name=spectator_camera)[::-1])
        if frames['research_spectator'].shape != (1080, 1920, 3):
            raise ValueError('Formal spectator video requires a native 1920x1080 render')
        for name, frame in frames.items():
            if name not in self.writers:
                self.writers[name] = self.writer_factory(str(self.output/'keyframes'/(name+'.mp4')),
                    fps=20, codec='libx264', macro_block_size=1,
                    pixelformat='yuv420p', ffmpeg_params=['-crf', '18', '-preset', 'veryfast', '-threads', '2'])
            self.writers[name].append_data(frame)
        self.index.write(json.dumps(dict(frame=self.frames, wall_seconds=captured_s,
                                        captured_utc=self.utc_clock(), scene_segment=segment,
                                        sequence=sequence, spectator_camera=spectator_camera))+'\n')
        self.index.flush()
        self.capture_times.append(captured_s)
        self.frames += 1
        if sequence is not None:
            self.last_sequence[segment] = sequence

    def _render_playback(self, elapsed_s):
        metadata = render_wall_time_playback(
            {name: self.output/'keyframes'/(name+'.mp4') for name in self.writers},
            self.capture_times, elapsed_s, self.output/RECORDING['playback_directory'],
            writer_factory=self.writer_factory, reader_factory=self.reader_factory)
        metadata['video_files'] = [RECORDING['playback_directory']+'/'+name for name in metadata['video_files']]
        return metadata

    def close(self):
        if self.closed:
            return
        self.closed = True
        encoding_started = self.clock()
        elapsed_s = encoding_started-self.started
        ended_utc = self.utc_clock()
        metadata = dict(**RECORDING, frames=self.frames, source_frames=self.frames,
            scene_segments=len(self.scenes), scope_started_utc=self.started_utc,
            scope_ended_utc=ended_utc, scope_wall_seconds=elapsed_s,
            first_observation_seconds=self.capture_times[0] if self.capture_times else None,
            last_observation_seconds=self.capture_times[-1] if self.capture_times else None,
            source_video_files=['keyframes/'+name+'.mp4' for name in self.writers],
            video_files=[], playback_status='pending')
        try:
            close_errors = []
            for writer in self.writers.values():
                try:
                    writer.close()
                except BaseException as exc:
                    close_errors.append(exc)
            if close_errors:
                raise close_errors[0]
            self.index.close()
            # This work is outside the measured camera/model timeline. It adds
            # no environment observations, renders, simulator steps or requests.
            try:
                from research.v4.development.confidence_gate_dev_20261006.phase_timing import phase
                encoding_phase = phase('human_video_postprocess', playback_speed=RECORDING['playback_speed'])
            except ImportError:
                encoding_phase = nullcontext()
            with encoding_phase:
                metadata.update(self._render_playback(elapsed_s), playback_status='completed')
        except BaseException as exc:
            metadata.update(playback_status='failed', playback_error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            metadata['postprocessing_wall_seconds'] = self.clock()-encoding_started
            if not self.index.closed:
                self.index.close()
            from research.v4.leash_v43.run_manifest import atomic_json
            atomic_json(self.output/'recording.json', metadata)
            derived = dict(metadata)
            derived['video_files'] = [Path(name).name for name in metadata['video_files']]
            derived['source_video_files'] = ['../'+name for name in metadata['source_video_files']]
            (self.output/RECORDING['playback_directory']).mkdir(exist_ok=True)
            atomic_json(self.output/RECORDING['playback_directory']/'recording.json', derived)


@contextmanager
def formal_video_scope(output, *, enabled):
    if not enabled:
        yield
        return
    from unittest.mock import patch
    from research.v3.bridge import libero_env
    recorder = SeparateViewRecorder(output)
    try:
        with patch.object(libero_env, 'PUBLIC_FRAME_SINK', recorder.capture):
            yield
    finally:
        original_error = sys.exc_info()[1]
        try:
            recorder.close()
        except BaseException as video_error:
            if original_error is None:
                raise
            # Preserve the actual episode failure (e.g. terminal provider 429).
            # recording.json independently records any postprocessing failure.
            if hasattr(original_error, 'add_note'):
                original_error.add_note('Human video postprocessing also failed: '
                    +type(video_error).__name__+': '+str(video_error))
