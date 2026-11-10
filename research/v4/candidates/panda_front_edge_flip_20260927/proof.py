"""Scoped original production factory with the own edge-flip geometry proof."""
import hashlib,json
from contextlib import contextmanager
from unittest.mock import patch
from research.v4.candidates.panda_local_cutout_native_20260927 import proof as previous
from .helper import HERE,ROOT
MjSim=previous.MjSim;create_scene=previous.create_scene;BASE_XML=previous.BASE_XML
PATCH_XML=(HERE/'candidate_scene.xml').read_text();EXPECTED=json.loads((HERE/'geometry_contract.json').read_text())
@contextmanager
def scoped():
    with patch.object(previous,'BASE_XML',BASE_XML),patch.object(previous,'PATCH_XML',PATCH_XML),patch.object(previous,'EXPECTED',EXPECTED),patch.object(previous,'create_scene',create_scene):yield
def semantic(before,after):
    with scoped():return previous.semantic(before,after)
def initialized(which,camera_observations,egl):
    with scoped():return previous.initialized(which,camera_observations,egl)
active_contacts=previous.active_contacts
def hashes():
    old=ROOT/'research/v4/candidates/panda_front_surface_remesh_20260927/source_freeze.json';pins=json.loads(old.read_text())
    for name,h in pins.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Parent122 drift '+name)
    files=[old,*HERE.glob('*.py'),HERE/'geometry_contract.json',HERE/'fixed_set_contract.json',HERE/'production_contract.json',
        HERE/'design.json',HERE/'flip_log.json',HERE/'front_surface_geometry.npz',HERE/'candidate_scene.xml',
        HERE/'link5_vis_0_edge_flip.obj',HERE/'HYPOTHESIS.md',HERE/'NOTICE.md',HERE/'HANDOFF.md',
        ROOT/'research/v4/candidates/panda_front_surface_remesh_20260927/native_pair_gpu1/report.json',
        ROOT/'research/v4/candidates/panda_front_shading_readonly_20260927/analysis.json']
    for p in files:pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
    if f.exists()and json.loads(f.read_text())!=pins:raise RuntimeError('Own freeze drift')
    return pins
