"""Freeze old qualified mechanics plus the new compiler-time visual overlay."""
import hashlib,json
from pathlib import Path
from .scene import HERE,ROOT
def hashes():
    old=ROOT/'research/v4/candidates/r4_05_live_rim_grasp_20260926/qualification_source_freeze33.json'
    pins=json.loads(old.read_text())
    for name,sha in pins.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('qualified mechanics drift: '+name)
    for p in [*HERE.glob('*.py'),HERE/'identity_card_uv.xml']:
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    f=HERE/'source_freeze.json'
    if f.exists()and pins!=json.loads(f.read_text()):raise ValueError('UV-overlay drift')
    return pins
