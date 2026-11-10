"""Opt-in release appearance at the existing compile seam; never auto-applied."""
from pathlib import Path
from contextvars import ContextVar
from threading import RLock
from unittest.mock import patch
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np,mujoco
from robosuite.utils.binding_utils import MjSim
from research.v4.candidates.panda_front_edge_flip_20260927 import helper as approved
from research.v4.candidates.panda_glyph_hide_visual_20260927 import helper as identity
from research.v4.candidates.panda_glyph_native_pair_20260927.native_pair import arrays,exact
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
APPEARANCE_VERSION='panda-original-MIT-plain-panel-142-optin-v1'
_SENTRY=ContextVar('safevla_release_panda_scope',default=None)
_LOCK=RLock()
_EXPECTED=json.loads((approved.HERE/'geometry_contract.json').read_text())

def source_identity(xml):
    """Exact source/unique body/material/geom identity before a single edit."""
    tree=ET.fromstring(xml)
    for mesh_name in ('robot0_link5_vis_0','robot0_link5_vis_1'):
        assets=[m for m in tree.findall('./asset/mesh')if m.get('name')==mesh_name]
        if len(assets)!=1:raise RuntimeError('Expected one original Panda mesh '+mesh_name)
        if assets[0].get('scale','1 1 1') not in ('1 1 1','1.0 1.0 1.0'):raise RuntimeError('Panda visual mesh scale changed')
    glyph=tree.find("./asset/mesh[@name='robot0_link5_vis_1']")
    try:h=hashlib.sha256(Path(glyph.get('file')).read_bytes()).hexdigest()
    except (OSError,TypeError)as e:raise RuntimeError('Original glyph source unavailable')from e
    if h!=identity.SHA:raise RuntimeError('Exact original glyph source changed')
    for name,material,geom_name in [('robot0_link5_vis_0','robot0_Shell001_001_001_003','robot0_g22_vis'),('robot0_link5_vis_1','robot0_Shell_001_001_003','robot0_g23_vis')]:
        geoms=[g for g in tree.findall('.//geom')if g.get('mesh')==name]
        if len(geoms)!=1:raise RuntimeError('Expected one Panda visual geom '+name)
        expected={'type':'mesh','contype':'0','conaffinity':'0','group':'1','material':material,'name':geom_name}
        if any(geoms[0].get(k)!=v for k,v in expected.items()):raise RuntimeError('Panda visual geom identity changed '+name)
        parent=tree.find(".//body[@name='robot0_link5']")
        if parent is None or geoms[0] not in list(parent):raise RuntimeError('Panda visual body identity changed '+name)
    # Approved helper validates exact original dark-panel SHA and142 outputSHA.
    return approved.patched_xml(xml)

def semantic(before,after):
    ba,aa=arrays(before),arrays(after);changed={k for k in ba if not exact(ba[k],aa[k])}
    if changed!=set(_EXPECTED['changed_arrays']):raise RuntimeError('Unexpected changed compiled fields '+str(sorted(changed^set(_EXPECTED['changed_arrays']))))
    gids={mujoco.mj_name2id(before,mujoco.mjtObj.mjOBJ_GEOM,n)for n in ('robot0_g22_vis','robot0_g23_vis')}
    if -1 in gids:raise RuntimeError('Expected Panda visual names missing')
    altered={int(i)for k in ('geom_pos','geom_quat','geom_size','geom_rbound','geom_aabb','geom_group')if k in changed for i in np.unique(np.argwhere(ba[k]!=aa[k])[:,0])}
    if altered!=gids:raise RuntimeError('Other geometry altered')
    glyph=mujoco.mj_name2id(before,mujoco.mjtObj.mjOBJ_GEOM,'robot0_g23_vis')
    if int(before.geom_group[glyph])!=1 or int(after.geom_group[glyph])!=5:raise RuntimeError('Wrong glyph display groups')
    if any(int(after.geom_contype[i])or int(after.geom_conaffinity[i])for i in gids):raise RuntimeError('Edited visual has collision enabled')
    collision=np.where((before.geom_contype!=0)|(before.geom_conaffinity!=0))[0]
    fields=['geom_type','geom_dataid','geom_bodyid','geom_contype','geom_conaffinity','geom_pos','geom_quat','geom_size','geom_rbound','geom_aabb','geom_friction','geom_solref','geom_solimp','geom_condim','geom_priority','geom_margin','geom_gap']
    if any(not exact(getattr(before,k)[collision],getattr(after,k)[collision])for k in fields):raise RuntimeError('Physical collision changed')
    unchanged=[]
    if before.nmesh!=after.nmesh:raise RuntimeError('Mesh inventory changed')
    for mid in range(before.nmesh):
        name=before.mesh(mid).name
        if name=='robot0_link5_vis_0':continue
        if name!=after.mesh(mid).name or identity.mesh_signature(before,mid)!=identity.mesh_signature(after,mid):raise RuntimeError('Other mesh changed '+str(name))
        unchanged.append(name)
    if any(not exact(v,arrays(after.opt)[k])for k,v in arrays(before.opt).items()):raise RuntimeError('Physics options changed')
    for k in dir(before.opt):
        v=getattr(before.opt,k)
        if isinstance(v,(int,float,bool))and v!=getattr(after.opt,k):raise RuntimeError('Physics option changed '+k)
    stat={}
    for k in ('center','extent','meaninertia','meanmass','meansize'):
        x,y=np.asarray(getattr(before.stat,k)),np.asarray(getattr(after.stat,k));stat[k]={'equal':exact(x,y),'before':x.tolist(),'after':y.tolist()}
        if k!='meansize'and not exact(x,y):raise RuntimeError('Model physical/camera scale stat changed '+k)
    return {'passed':True,'numeric_model_arrays':len(ba),'intentional_visual_changed_fields':sorted(changed),'unchanged_numeric_fields':len(ba)-len(changed),
            'edited_visual_geom_ids':sorted(gids),'collision_geometries':len(collision),'collision_fields_exact':fields,
            'unchanged_mesh_count':len(unchanged),'unchanged_mesh_names':unchanged,'physical_options_exact':True,'statistics':stat}

def construct(factory,*args,appearance_version=None,**kwargs):
    """Default factory is untouched; explicit version scopes one precompile edit.

    No installed/shared file mutation, extra control, forward, or state restore.
    ContextVar isolates unrelated thread calls during the short compile scope.
    The original descriptor is restored even if construction/source checks fail.
    """
    if appearance_version is None:return factory(*args,**kwargs)
    if appearance_version!=APPEARANCE_VERSION:raise ValueError('Unknown release appearance version')
    if _SENTRY.get()is not None:raise RuntimeError('Nested appearance factory unsupported')
    with _LOCK:
        old_descriptor=MjSim.__dict__['from_xml_string'];original=MjSim.from_xml_string
        context={'calls':[]};token=_SENTRY.set(context)
        def compiled(cls,incoming):
            if _SENTRY.get()is not context:return original(incoming)
            candidate=source_identity(incoming)
            # Stateless actual precompile guard for this task's complete model.
            before=mujoco.MjModel.from_xml_string(incoming);after=mujoco.MjModel.from_xml_string(candidate)
            guard=semantic(before,after)
            context['calls'].append({'original_xml_sha256':hashlib.sha256(incoming.encode()).hexdigest(),
                                      'candidate_xml_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'guard':guard})
            return original(candidate)
        try:
            with patch.object(MjSim,'from_xml_string',classmethod(compiled)):scene=factory(*args,**kwargs)
        finally:
            _SENTRY.reset(token)
            if MjSim.__dict__['from_xml_string']is not old_descriptor:raise RuntimeError('Production compile descriptor not restored')
        if len(context['calls'])!=1:
            if hasattr(scene,'close'):scene.close()
            raise RuntimeError('Expected exactly one original production compilation')
        scene.release_appearance_version=APPEARANCE_VERSION
        scene.release_appearance_metadata={'version':APPEARANCE_VERSION,'opt_in':True,'compiled_calls':context['calls'],
            'visual_mesh_source':str(approved.HERE/'link5_vis_0_edge_flip.obj'),
            'visual_mesh_sha256':hashlib.sha256((approved.HERE/'link5_vis_0_edge_flip.obj').read_bytes()).hexdigest(),
            'notice':str(approved.HERE/'NOTICE.md'),'historical_results_upgraded':False,'online_protocol_or_task_changed':False}
        return scene
