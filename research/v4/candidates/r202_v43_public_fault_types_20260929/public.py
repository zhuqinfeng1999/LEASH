"""Source-bound catch repair, retaining the own reader's successful recipe."""
import ast
from collections.abc import Mapping
from types import FunctionType

import numpy as np

from research.v4.accelerated.observation import _camera, PUBLIC_CAMERAS
from research.v4.candidates.r202_v43_actual_consumer_20260929 import runtime as parent
from research.v4.candidates.r4_02_typed_public_masks_20260927.public import Client, DetectionMiss


class PublicReaderTechnicalFault(Exception):
    """A typed acquisition fault, never a semantic detection miss."""
    def __init__(self, row):
        self.row = row
        super().__init__("Public reader technical acquisition fault")


def _fault(reader, exc, **metadata):
    row = {**metadata, "fault": True, "failure_kind": "public_reader_technical_fault",
           "error_type": type(exc).__name__, "error": str(exc)}
    reader.log.append(row)
    raise PublicReaderTechnicalFault(row) from exc


def _capture(reader):
    # The callback is the original capture_agent; exactly one acquisition.
    try:
        packet = reader.scene.capture_agent()
    except Exception as exc:
        _fault(reader, exc, stage="capture")
    try:
        if not isinstance(packet, Mapping):
            raise TypeError("Public packet must be a mapping")
        frame = packet["frame_id"]
        if type(frame) is not int or frame < 0:
            raise ValueError("Public frame identifier malformed")
        if not isinstance(packet["cameras"], Mapping) or set(packet["cameras"]) != set(PUBLIC_CAMERAS):
            raise ValueError("Exactly the two declared public cameras required")
        for name in PUBLIC_CAMERAS:
            camera = packet["cameras"][name]
            _camera(name, camera, frame)  # Reuse current metric image/calibration validator.
            if abs(np.linalg.det(np.asarray(camera["extrinsics"], float)[:3, :3]) - 1) > 1e-4:
                raise ValueError("Public camera rotation malformed")
        eef = np.asarray(packet["robot_state"]["eef_pos"], float)
        if eef.shape != (3,) or not np.isfinite(eef).all():
            raise ValueError("Required public EEF position malformed")
        # No latch, object axis, wrist quaternion or full gripper state is read.
    except Exception as exc:
        _fault(reader, exc, stage="public_packet")
    return packet


def _metadata(packet, camera_name, name, prompt):
    return {"frame_id": packet["frame_id"], "camera": camera_name,
            "name": "item" if name == "iron" else name, "prompt": prompt}


def _segment(reader, packet, camera_name, name, prompt, image):
    metadata = _metadata(packet, camera_name, name, prompt)
    # Use the reviewed real Sam3Result/DetectionMiss rules; no R402 provider.
    result = Client(reader.sam, metadata).segment(image, text_prompt=prompt, min_score=.2)
    # The actual Sam3Client decoder emits a Boolean mask and declared shape.
    # Invalid success data is a technical fault, never missing geometry.
    mask = np.asarray(result.mask)
    if (mask.dtype != np.bool_ or mask.shape != (480, 640)
            or result.mask_shape != (480, 640)):
        raise ValueError("SAM3 successful mask schema malformed")
    return result


def _view_fault(reader, packet, camera_name, name, prompt, exc):
    _fault(reader, exc, **_metadata(packet, camera_name, name, prompt),
           stage="per_view_acquisition")


def public_reader_class():
    original_reader = parent.public_reader_class()
    tree = ast.parse(original_reader.selected_acquisition_source)
    captures = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and ast.unparse(n) == "self.scene.capture_agent()"]
    segments = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "self.sam.segment"]
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)
                and ast.unparse(n.type) == "Exception"]
    if len(captures) != 1 or len(segments) != 1 or len(handlers) != 1:
        raise RuntimeError("Exact source-bound capture/segment/catch seam changed")
    if ast.unparse(segments[0]) != "self.sam.segment(buffer.getvalue(), text_prompt=prompt, min_score=0.2)":
        raise RuntimeError("Original SAM call parameters changed")
    if ast.unparse(handlers[0]) != (
            "except Exception as exc:\n    self.log.append({'frame_id': packet['frame_id'], "
            "'camera': camera_name, 'name': name, 'error': str(exc)})"):
        raise RuntimeError("Original per-view broad catch changed")
    captures[0].func = ast.Name("_capture", ast.Load())
    captures[0].args = [ast.Name("self", ast.Load())]
    segments[0].func = ast.Name("_segment", ast.Load())
    segments[0].args = [ast.Name(x, ast.Load()) for x in
                        ("self", "packet", "camera_name", "name", "prompt")] + segments[0].args
    segments[0].keywords = []
    miss = ast.parse("""try:
    pass
except DetectionMiss as exc:
    self.log.append({**exc.row, 'failure_kind': 'structured_public_detection_miss', 'fault': False})
except Exception as exc:
    _view_fault(self, packet, camera_name, name, prompt, exc)
""").body[0].handlers
    owner = next(n for n in ast.walk(tree) if isinstance(n, ast.Try) and handlers[0] in n.handlers)
    owner.handlers = miss
    acquisition = original_reader.__bases__[0].__call__
    namespace = dict(acquisition.__globals__, _capture=_capture, _segment=_segment,
                     _view_fault=_view_fault, DetectionMiss=DetectionMiss)
    exec(compile(ast.fix_missing_locations(tree), __file__ + "::selected-acquisition", "exec"), namespace)
    base = type("OwnR202TypedAcquisition", (parent.acquisition.PublicWorkspaceReader,),
                {"__call__": namespace["__call__"]})
    reader = type("OwnR202TypedPublicReader", (base,), {})
    outer = parent.own.PublicWorkspaceReader.__call__
    reader.__call__ = FunctionType(outer.__code__, outer.__globals__, outer.__name__,
                                   outer.__defaults__, (parent._cell(reader),))
    reader.locate_vla_target = parent.own.PublicWorkspaceReader.locate_vla_target
    reader.original_outer_code = outer.__code__
    reader.selected_acquisition_source = ast.unparse(tree)
    reader.transform_counts = {"capture": 1, "segment": 1, "broad_catch": 1}
    return reader
