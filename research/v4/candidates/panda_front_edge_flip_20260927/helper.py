"""Same original XML repair except fixed-set triangulation, scoped only."""
import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
from research.v4.candidates.panda_front_surface_remesh_20260927 import helper as parent
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def patched_xml(original):
    proof=json.loads((HERE/'design.json').read_text());replacement=HERE/'link5_vis_0_edge_flip.obj'
    if not proof['passed']or hashlib.sha256(replacement.read_bytes()).hexdigest()!=proof['output_sha256']:raise RuntimeError('Edge-flip source drift')
    tree=ET.fromstring(parent.patched_xml(original));mesh=tree.find("./asset/mesh[@name='robot0_link5_vis_0']")
    if mesh is None or Path(mesh.get('file')).resolve()!=parent.HERE/'link5_vis_0_smooth_front.obj':raise RuntimeError('Expected exact parent patch')
    mesh.set('file',str(replacement));return ET.tostring(tree,encoding='unicode')
