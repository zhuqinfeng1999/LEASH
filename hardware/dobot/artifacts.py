"""Public hardware artifact store implementing RPent's localization contract."""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image


class HardwareArtifacts:
    def __init__(self, driver, output):
        self.driver, self.output = driver, Path(output)
        self.records = []

    def append(self, data):
        observation = data["observation"]
        index = len(self.records)
        images = dict(
            wrist=observation["wrist_image"], agentview=observation["observer_image"]
        )
        artifacts = {camera + ".png": path for camera, path in images.items()}
        for camera, path in images.items():
            artifacts[camera + "_high.png"] = path
        for camera, key in (('wrist', 'wrist'), ('agentview', 'third_person')):
            meta = data['public_metadata']['trial_geometry']['camera_meta'].get(key, {})
            if meta.get('depth_path') and (camera == 'wrist' or meta.get('metric_projection_available')):
                for suffix in ("_world.npz", "_world_high.npz", "_depth.npz"):
                    artifacts[camera + suffix] = "recorded_sdk_map"
        artifacts.update({camera + "_metadata.json": camera for camera in images})
        self.records.append(
            SimpleNamespace(step_idx=index, artifacts=artifacts, data=data, extra={})
        )

    def get(self, step=-1):
        # Match EnvState: only -1 means latest. Python's other negative
        # indices must not silently substitute an older camera/depth frame.
        if step == -1:
            if not self.records:
                raise LookupError('no steps available')
            step = len(self.records) - 1
        if step is None or step < 0:
            raise ValueError('step must be -1 or nonnegative')
        if 0 <= step < len(self.records):
            return self.records[step]
        raise KeyError(f'step {step} not present in state trace')

    def annotate_projection(self, result, **arguments):
        """Attach the same measured provenance to live and recorded projections."""
        result = dict(result)
        key = None
        try:
            if result.get('error') and 'row' in arguments and 'col' in arguments:
                key = 'depth_diagnostic'
                result[key] = self.depth_diagnostic(**arguments)
            elif not result.get('error'):
                key = 'measurement_diagnostic'
                result[key] = self.projection_diagnostic(result, **arguments)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            result[key] = dict(reason='diagnostic_unavailable', error=str(exc))
        return result

    def depth_diagnostic(self, row, col, *, camera='wrist', step=-1, **_kwargs):
        """Report raw versus calibration-scope failure for the exact saved pixel."""
        if camera not in ('wrist', 'agentview', 'third_person'):
            return dict(reason='unknown_camera', camera=camera)
        if type(step) is not int or (step != -1 and not 0 <= step < len(self.records)) or not self.records:
            return dict(reason='recorded_step_unavailable', step=step)
        record = self.get(step)
        key = 'third_person' if camera in ('agentview', 'third_person') else 'wrist'
        meta = record.data['observation']['trial_geometry']['camera_meta'].get(key, {})
        if not meta.get('depth_path'):
            return dict(reason='recorded_depth_unavailable', camera=camera)
        import importlib.util
        source = Path(__file__).with_name('scoped_depth.py')
        spec = importlib.util.spec_from_file_location('dobot_depth_diagnostic', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.depth_diagnostic(np.load(meta['depth_path']) * meta['depth_scale_m'], meta, row, col)
        result.update(step=record.step_idx, camera=camera)
        # Supply measured neighbours in the same response, not an invented
        # replacement for the failed pixel. This saves another model/tool turn
        # while leaving surface selection to the planner's RGB inspection.
        map_name = ('wrist' if camera == 'wrist' else 'agentview') + '_world_high.npz'
        candidates = result.get('nearby_measured_pixels', [])
        if candidates and self.exists(map_name, step=record.step_idx):
            try:
                world = self.load(map_name, step=record.step_idx)
                for candidate in candidates:
                    point = np.asarray(world[candidate['row'], candidate['col']], dtype=float)
                    if point.shape == (3,) and np.isfinite(point).all() and np.any(point != 0):
                        candidate.update(world_xyz=point.tolist(), coordinate_frame='dobot_base',
                                         surface_verified=None)
                result['candidate_coordinates_scope'] = (
                    'Each XYZ belongs only to its listed measured row/col in this recorded step. '
                    'It is not the requested pixel, a verified object point, or a grasp pose.')
            except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
                result['candidate_projection_error'] = str(exc)
        return result

    def exists(self, name, *, step=-1):
        return name in self.get(step).artifacts

    def projection_diagnostic(self, result, *, camera='agentview', step=-1, row=None, col=None, **_kwargs):
        """Attach measurement provenance without changing/rejecting a valid XYZ.

        A successful array lookup is not an independent surface validation.
        In particular, reflective surfaces can return finite erroneous depth.
        """
        if camera not in ('wrist', 'agentview'):
            raise ValueError('Unknown camera')
        record = self.get(step)
        geometry = record.data['observation']['trial_geometry']
        key = 'third_person' if camera == 'agentview' else 'wrist'
        meta = geometry['camera_meta'][key]
        value = dict(xyz_unchanged=True, surface_verified=None,
                     surface_verification_status='not_independently_assessed',
                     point_is_grasp_or_placement_pose=False)
        if type(row) is int and type(col) is int:
            raw = np.load(meta['depth_path']) * meta['depth_scale_m']
            if not 0 <= row < raw.shape[0] or not 0 <= col < raw.shape[1]:
                raise ValueError('Pixel outside image')
            z = float(raw[row, col])
            value['raw_depth_m'] = z if np.isfinite(z) else None
            fit = meta.get('local_depth_correction')
            within = bool(fit and fit['valid_raw_depth_range_m'][0] <= z <= fit['valid_raw_depth_range_m'][1])
            value['correction_status'] = 'calibrated_correction' if within else 'uncorrected_raw_depth'
            patch = raw[max(0,row-2):row+3, max(0,col-2):col+3]
            measured = patch[np.isfinite(patch) & (patch > 0) & (patch < 10)]
            value['patch_valid_pixels'] = int(measured.size)
            value['patch_depth_range_m'] = [float(measured.min()),float(measured.max())] if measured.size else None
        point = result.get('world_xyz', result.get('median_xyz'))
        plane = geometry.get('measured_table_plane')
        if point is not None and plane is not None:
            value['signed_distance_to_measured_table_m'] = float(
                np.asarray(point) @ np.asarray(plane['normal']) + plane['offset_m'])
            value['table_distance_meaning'] = 'Positive is above the measured plane. This diagnostic does not replace depth, infer object height, or block a command.'
        return value

    def load(self, name, *, step=-1):
        record = self.get(step)
        if name in record.extra:
            return record.extra[name]
        if name not in record.artifacts:
            raise FileNotFoundError(name)
        if name.endswith("_metadata.json"):
            camera = record.artifacts[name]
            key = "third_person" if camera == "agentview" else "wrist"
            meta = record.data["public_metadata"]["trial_geometry"]["camera_meta"][key]
            intr = meta.get('intrinsics', {})
            return dict(
                meta,
                image_size=[intr.get('height'), intr.get('width')],
                eef_reference="grip_site",
            )
        if name.endswith("_world.npz") or name.endswith("_world_high.npz"):
            if not hasattr(self.driver, "recorded_world_map"):
                raise ValueError("Hardware SDK world-map reader unavailable")
            if name.startswith('agentview_'):
                return self.driver.recorded_world_map(record.data["observation"], camera='third_person')
            return self.driver.recorded_world_map(record.data["observation"])
        if name in ('wrist_depth.npz', 'agentview_depth.npz'):
            key = 'third_person' if name.startswith('agentview_') else 'wrist'
            meta = record.data["observation"]["trial_geometry"]["camera_meta"][key]
            # Use the same correction as SDK world maps; otherwise the low-
            # resolution reader would disagree with the high-resolution one.
            import importlib.util
            source = Path(__file__).with_name('scoped_depth.py')
            spec = importlib.util.spec_from_file_location('dobot_scoped_depth', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.corrected_depth(np.load(meta["depth_path"]) * meta["depth_scale_m"],meta)
        if name.endswith(".png"):
            with Image.open(record.artifacts[name]) as im:
                return np.asarray(im.convert("RGB")).copy()
        raise ValueError("Unsupported hardware artifact")

    def load_bytes(self, name, *, step=-1):
        artifacts = self.get(step).artifacts
        if name in artifacts:
            return Path(artifacts[name]).read_bytes()
        # Reference segment() returns the generated overlay path to this reader.
        # Only paths already registered for this step are permitted.
        target = Path(name).resolve()
        if any(Path(value).resolve() == target for value in artifacts.values()):
            return target.read_bytes()
        raise FileNotFoundError("Unregistered artifact")

    def save(self, name, value, *, step=-1):
        # Frozen segment() supplies generated basenames, never planner paths.
        if Path(name).name != name:
            raise ValueError("Artifact name must be a basename")
        self.get(step).extra[name] = value
        self.get(step).artifacts[name] = str(
            self.output / f"step_{self.get(step).step_idx:04d}" / name
        )
        path = Path(self.get(step).artifacts[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".png"):
            Image.fromarray(value).save(path)
        elif name.endswith(".json"):
            import json

            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return str(path)
