"""Opt-in public interface only; original constructor and generic PROCEED block stay."""
from pathlib import Path
import hashlib,json
from research.v4.candidates.r4_normal_crib_method_optin_20260928 import binding as normal
from research.v4.candidates.r3_v42_tabletop_20260925.scene import R3TabletopFacade
from research.v4.candidates.r4_02_proceed_outlet_20260927.scene import Scene as PublicLatchScene
from research.v4.candidates.r4_02_proceed_outlet_20260927.public import task_evidence

HERE=Path(__file__).resolve().parent;ROOT=normal.ROOT
VERSION='R4-normal-own-full-public-packet-latch-optin-v1-20260928'
INSTANCES=('R4-01','R4-03','R4-04','R4-05','R4-06')

def scene_class(instance):
    if instance not in INSTANCES:raise ValueError('Exact five generic normal instances only')
    guarded=normal.scene_class(instance)
    own_capture=guarded.__bases__[0].capture_agent
    if own_capture is not R3TabletopFacade.capture_agent:
        raise RuntimeError('Own existing full public capture identity changed')
    class FullPublicScene(guarded):
        # Inherit guarded __init__ byte-for-byte: own instance/reset/source XML.
        capture_agent=own_capture
        public_grasp_latched=PublicLatchScene.public_grasp_latched
        public_sensor_revision=PublicLatchScene.public_sensor_revision
        public_packet_binding_version=VERSION
    FullPublicScene.__name__=instance.replace('-','')+'OwnFullPublicScene'
    return FullPublicScene

def hashes():
    pins=json.loads((HERE/'source_freeze.json').read_text())['sha256']
    changed=[n for n,v in pins.items()if hashlib.sha256((Path(n)if n.startswith('/')else ROOT/n).read_bytes()).hexdigest()!=v]
    if changed:raise RuntimeError('Public packet binding source drift: '+repr(changed))
    return pins

def runner_module(instance):
    selected=scene_class(instance);m=normal.runner_module(instance)
    # These are the actual generated main's selected globals, not unused aliases.
    if m.main.__globals__ is not m.__dict__:raise RuntimeError('Generated main namespace changed')
    if 'Scene'not in m.main.__code__.co_names or 'evidence_from_scene'not in m.main.__code__.co_names:
        raise RuntimeError('Actual public factory/evidence consumer anchors changed')
    m.OWN_SCENE=selected;m.Scene=selected
    m.evidence_from_scene=task_evidence  # Existing minimal view -> original strict helper.
    m.OWN_HASHES=hashes;m.PUBLIC_PACKET_BINDING_VERSION=VERSION
    return m
