"""Thin source-versioned hooks over the immutable six-instance injection recipe."""
import ast,hashlib,json
from pathlib import Path
from research.v4.candidates.e1_plain_config_dispatch_20260927 import hooks as original
from .factory import components
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PACKAGE='research.v4.candidates.e1_container_unknown_tilt_dispatch_20260927'
VERSION='e1-own-container-ordinary-rpent-unknown-tilt-failclosed-v42-candidate-v2'

def source_hashes():
    pins=dict(original.source_hashes())
    old=Path(original.HERE)/'source_freeze.json';pins[str(old.relative_to(ROOT))]=hashlib.sha256(old.read_bytes()).hexdigest()
    gap=ROOT/'research/v4/candidates/e1_container_tilt_gap_20260927'
    for p in [*HERE.glob('*.py'),*HERE.glob('*.md'),*[p for p in HERE.glob('*.json')if p.name!='source_freeze.json'],gap/'HANDOFF.md',gap/'source_freeze.json']:
        pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    pins=dict(sorted(pins.items()));f=HERE/'source_freeze.json'
    if f.exists()and json.loads(f.read_text())!=pins:raise RuntimeError('Own unknown-tilt dispatch source drift')
    return pins

def runner_module(instance):
    # Original functions run in an isolated namespace. Parent globals remain untouched.
    path=Path(original.__file__);text=path.read_text();tree=ast.parse(text)
    ns=dict(vars(original),HERE=HERE,ROOT=ROOT,PACKAGE=PACKAGE,VERSION=VERSION,
            components=components,source_hashes=source_hashes)
    for name in ('configure','runner_module'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
        exec(compile(ast.get_source_segment(text,node),str(path)+' [container-v2 isolated]','exec'),ns)
    return ns['runner_module'](instance)
