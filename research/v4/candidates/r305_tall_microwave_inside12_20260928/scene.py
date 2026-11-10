"""Normal opt-in R305 constructor: same F/PARK/cart, visible ear and tall device."""
from dataclasses import replace as original_replace
from pathlib import Path
from types import FunctionType
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from research.v4.candidates.r305_front_source_material_copy_20260928 import scene as front
from research.v4.candidates.r305_tall_microwave_single_lug_20260928 import check as geometry

H=Path(__file__).resolve().parent
P=H.parent/'r305_tall_microwave_single_lug_20260928'
EAR=H.parent/'r305_single_ceramic_lug_20260928/assets/fork_bowl_single_lug.xml'
ENV=front.parent.parent.OwnCartEnv
ORIGINAL_LOADER=ENV._load_model
ORIGINAL_FRONT_INIT=front.FrontSourceMaterialScene.__init__


def same(key,a,b):
    if key in ('pos','size','quat','euler','scale'):
        return bool(np.array_equal(np.fromstring(a,sep=' '),np.fromstring(b,sep=' ')))
    if key=='file':return Path(a).resolve()==Path(b).resolve()
    return a==b


def qualified_device_delta(root):
    before=ET.parse(H.parent/'r305_single_ceramic_lug_20260928/full_scene_lug.xml').getroot()
    after=ET.parse(P/'full_scene_tall_lug.xml').getroot()
    old={(e.tag,e.get('name')):e for e in before.iter()if e.get('name')}
    new={(e.tag,e.get('name')):e for e in after.iter()if e.get('name')}
    receipts=[]
    for identity,e in new.items():
        if identity not in old:raise RuntimeError('Qualified model topology delta unexpected')
        prior=old[identity]
        for key,value in e.attrib.items():
            if key=='name' or value==prior.get(key):continue
            if not str(identity[1]).startswith('microwave_1_'):raise RuntimeError('Unexpected non-device XML delta')
            candidates=[n for n in root.iter(e.tag)if n.get('name')==identity[1]]
            if len(candidates)!=1:raise RuntimeError('Actual precompile named device seam differs '+str(identity))
            node=candidates[0]
            if node.get(key) is None or not same(key,node.get(key),prior.get(key)):
                raise RuntimeError('Actual precompile device value differs '+str(identity)+'/'+key)
            receipts.append({'tag':e.tag,'name':identity[1],'field':key,'before':node.get(key),'after':value})
            node.set(key,value)
    if not receipts:raise RuntimeError('No qualified taller-device delta consumed')
    return receipts


class TallLugScene(front.FrontSourceMaterialScene):
    layout_version='R305-F-source-real-positive-ear-tall350-cavity-inside12-expert-v1'
    common_reset_controls=20

    def __init__(self,*args,**kwargs):
        if ENV._load_model is not ORIGINAL_LOADER or front.replace is not original_replace:
            raise RuntimeError('Original constructor scope already changed')
        self.tall_precompile_receipts=[];ear_count=[0]
        def replace_binding(binding,**kw):
            if binding.name=='fork_bowl_1':
                if tuple(binding.grasp_local)!=(.059,0.,.022):raise RuntimeError('Original own zone differs')
                ear_count[0]+=1
                kw.update(xml=str(EAR),grasp_local=(.080,0.,.022))
            return original_replace(binding,**kw)
        def load(env):
            ORIGINAL_LOADER(env)
            self.tall_precompile_receipts.append(qualified_device_delta(env.model.root))
        ns=dict(ORIGINAL_FRONT_INIT.__globals__);ns['replace']=replace_binding
        init=FunctionType(ORIGINAL_FRONT_INIT.__code__,ns,ORIGINAL_FRONT_INIT.__name__,
                          ORIGINAL_FRONT_INIT.__defaults__,ORIGINAL_FRONT_INIT.__closure__)
        with patch.object(ENV,'_load_model',load):init(self,*args,**kwargs)
        if ear_count!=[1]or len(self.tall_precompile_receipts)!=1:raise RuntimeError('Own normal constructor binding count differs')
        if ENV._load_model is not ORIGINAL_LOADER or front.replace is not original_replace:
            raise RuntimeError('Normal constructor scope not restored')
        self.appearance_asset_scope='Original Apache tall device + unchanged authored CC0 positive lug'
        self.constructor_binding_restored=True
        self.settling_trace=[]
        sim=self.backend.sim;m,d=sim.model._model,sim.data._data
        item=m.body('fork_bowl_1_main').id;oven=m.body('microwave_1_main').id
        self.settling_before={'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),'time':float(d.time),
                              'sequence':self.sequence,'payload':d.xpos[item].tolist(),
                              'payload_quat':d.xquat[item].tolist(),'oven':d.xpos[oven].tolist()}
        for i in range(self.common_reset_controls):
            self.step(np.asarray([0.,0.,0.,0.,0.,0.,-1.],dtype=np.float32))
            cs=geometry.full_contacts(m,d)
            row={'index':i+1,'time':float(d.time),'sequence':self.sequence,'qpos':d.qpos.tolist(),
                 'qvel':d.qvel.tolist(),'payload':d.xpos[item].tolist(),'payload_quat':d.xquat[item].tolist(),
                 'oven':d.xpos[oven].tolist(),'contacts':cs,'assist_events':list(self.assist_events)}
            self.settling_trace.append(row)
            if any(c['hard_stop']for c in cs):raise RuntimeError('First real20-neutral hard contact')
        # This is the declared common factory reset, not an expert-only action.
        # sequence remains20 so future per-episode1200 accounting includes it.
