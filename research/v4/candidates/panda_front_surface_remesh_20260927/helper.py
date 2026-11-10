"""Scoped candidate XML substitution; no installed or common asset changes."""
import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]

def patched_xml(original):
    proof=json.loads((HERE/'design.json').read_text()); source=ROOT/proof['source']; replacement=HERE/'link5_vis_0_smooth_front.obj'
    if not proof['passed'] or hashlib.sha256(source.read_bytes()).hexdigest()!=proof['source_sha256'] or hashlib.sha256(replacement.read_bytes()).hexdigest()!=proof['output_sha256']:
        raise RuntimeError('Front repair source drift')
    tree=ET.fromstring(original)
    meshes=[x for x in tree.findall('./asset/mesh')if x.get('name')=='robot0_link5_vis_0']
    glyphs=[x for x in tree.findall('.//geom')if x.get('mesh')=='robot0_link5_vis_1']
    if len(meshes)!=1 or len(glyphs)!=1: raise RuntimeError('Expected exact unique panel and glyph')
    glyph=glyphs[0]
    if any(glyph.get(k)!=v for k,v in {'type':'mesh','contype':'0','conaffinity':'0','group':'1','material':'robot0_Shell_001_001_003'}.items()):
        raise RuntimeError('Glyph visual identity changed')
    if hashlib.sha256(Path(meshes[0].get('file')).read_bytes()).hexdigest()!=proof['source_sha256']:
        raise RuntimeError('Installed panel identity changed')
    meshes[0].set('file',str(replacement));glyph.set('group','5')
    return ET.tostring(tree,encoding='unicode')
