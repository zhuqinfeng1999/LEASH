"""One explicit, opt-in release appearance constructor; no execution changes."""
from pathlib import Path
from contextvars import ContextVar
from threading import RLock
from unittest.mock import patch
import hashlib,xml.etree.ElementTree as ET
import mujoco
from robosuite.utils.binding_utils import MjSim
from research.v4.candidates.panda_release_factory_optin_20260927 import helper as panda
from research.v4.candidates.garment_plain_cotton_material_20260927 import helper as cotton
from research.v4.candidates.garment_plain_cotton_material_20260927.check_cpu import semantic as cotton_semantic
from research.v4.candidates.panda_glyph_native_pair_20260927.native_pair import arrays,exact
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
APPEARANCE_VERSION='release-Panda142-cotton82-authoredS101232-optin-v1'
_SENTRY=ContextVar('safevla_release_composed_scope',default=None);_LOCK=RLock()
KNOWN_PERSON_SKINS={'seated_person_body',cotton.NAME,'seated_person_short04','seated_person_high-poly'}

def garment_identity(xml):
    """No-person skip before any garment asset access; unknown person fails."""
    tree=ET.fromstring(xml);skins=tree.findall('./asset/skin');names=[s.get('name')for s in skins]
    has_person=any((b.get('name')or'').startswith('person_')for b in tree.findall('.//body'))or any((n or'').startswith('seated_person_')for n in names)
    if not has_person:return {'status':'explicit_skip_no_person','garment_asset_validation_required':False,'mapping':None}
    if len(names)!=len(set(names))or set(names)!=KNOWN_PERSON_SKINS:raise RuntimeError('Unknown or incomplete selected person garment/skin inventory')
    skin=tree.find(f"./asset/skin[@name='{cotton.NAME}']")
    if skin.get('material')!=cotton.NAME+'_material'or skin.get('group')!='1'or skin.get('rgba')!='1 1 1 1':raise RuntimeError('Selected garment visual identity changed')
    if len(skin.get('vertex','').split())!=4035*3:raise RuntimeError('Selected garment vertex inventory changed')
    textures=[t for t in tree.findall('./asset/texture')if t.get('name')==cotton.NAME+'_texture']
    if len(textures)!=1 or textures[0].get('type')!='2d':raise RuntimeError('Selected garment texture identity changed')
    tex=Path(textures[0].get('file',''))
    if tex.resolve()!=(cotton.SOURCE/'male_casualsuit06_diffuse.png').resolve():raise RuntimeError('Unknown selected garment texture source')
    cotton.check_source()
    return {'status':'exact_selected_garment','garment_asset_validation_required':True,'mapping':None}

def transform(xml):
    identity=garment_identity(xml)
    middle=panda.source_identity(xml)
    if identity['status']=='explicit_skip_no_person':return middle,middle,identity
    candidate,mapping=cotton.patched_xml(middle);identity['mapping']=mapping
    return middle,candidate,identity

def semantic(original,panda_only,candidate,identity):
    p=panda.semantic(original,panda_only)
    c=None
    if identity['status']=='exact_selected_garment':c=cotton_semantic(panda_only,candidate,identity['mapping'])
    elif identity['status']=='explicit_skip_no_person':
        if any(not exact(v,arrays(candidate)[k])for k,v in arrays(panda_only).items()):raise RuntimeError('No-person material stage modified model')
    else:raise RuntimeError('Unknown garment transform status')
    a,b=arrays(original),arrays(candidate);changed=sorted(k for k in a if not exact(a[k],b[k]))
    permitted=set(p['intentional_visual_changed_fields'])|(set(c['changed_model_arrays'])if c else set())
    if set(changed)-permitted:raise RuntimeError('Composed nonvisual array changed')
    return {'panda142_guard':p,'garment82_guard':c,'garment_status':identity['status'],
            'total_numeric_arrays':len(a),'actual_composed_changed_fields':changed,
            'unchanged_numeric_arrays':len(a)-len(changed),
            'each_visual_transform_nonvisual_semantic_invariance':True,
            'collision_mass_inertia_camera_controller_options_exact':True}

def construct(factory,*args,appearance_version=None,**kwargs):
    """One scoped precompile transform, exact existing factory otherwise.

    Same common entry is used for the authored S101, E102 and no-person R2.
    Factory-specific layout/assets/controller remain the caller's own choices.
    Provenance attributes are offline metadata; not online decision inputs.
    """
    if appearance_version is None:return factory(*args,**kwargs)
    if appearance_version!=APPEARANCE_VERSION:raise ValueError('Unknown composed release appearance version')
    if _SENTRY.get()is not None:raise RuntimeError('Nested composed appearance factory unsupported')
    with _LOCK:
        descriptor=MjSim.__dict__['from_xml_string'];original=MjSim.from_xml_string
        context={'calls':[]};token=_SENTRY.set(context)
        def compiled(cls,xml):
            if _SENTRY.get()is not context:return original(xml)
            middle,candidate,identity=transform(xml)
            m0=mujoco.MjModel.from_xml_string(xml);mp=mujoco.MjModel.from_xml_string(middle)
            mc=mp if candidate==middle else mujoco.MjModel.from_xml_string(candidate)
            guard=semantic(m0,mp,mc,identity)
            context['calls'].append({'original_xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
                'panda_only_xml_sha256':hashlib.sha256(middle.encode()).hexdigest(),
                'composed_xml_sha256':hashlib.sha256(candidate.encode()).hexdigest(),
                'garment_identity':identity,'guard':guard})
            return original(candidate)
        try:
            with patch.object(MjSim,'from_xml_string',classmethod(compiled)):scene=factory(*args,**kwargs)
        finally:
            _SENTRY.reset(token)
            if MjSim.__dict__['from_xml_string']is not descriptor:raise RuntimeError('Original production descriptor not restored')
        if len(context['calls'])!=1:
            if hasattr(scene,'close'):scene.close()
            raise RuntimeError('Expected one own production compilation')
        scene.release_appearance_version=APPEARANCE_VERSION
        scene.release_appearance_metadata={'version':APPEARANCE_VERSION,'opt_in':True,'compile_calls':context['calls'],
            'default_or_historical_adopted':False,'online_fields_added':False,'task_physics_replaced':False,
            'panda_notice':str(panda.approved.HERE/'NOTICE.md'),'garment_notice':str(cotton.HERE/'NOTICE.md'),
            'limitations':['Tiny unchanged denim label remains unverified; not all-human clearance.',
                'Combined appearance needs root native review before common/formal adoption.']}
        return scene
