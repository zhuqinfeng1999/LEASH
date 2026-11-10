"""Exactly two source support pairs, original80 OPEN startup and all hard risk."""
import json
from pathlib import Path
from types import FunctionType
from xml.etree import ElementTree as ET
import mujoco
import numpy as np
from research.v4.candidates.r302_front_slot_fixed80_reset_20260928 import scene as old
from .admit import verify

H=Path(__file__).resolve().parent
PARAMS=json.loads((H/'old_effective_contacts.json').read_text())
def text_vector(v):return ' '.join(format(float(x),'.17g') for x in v)

class SourcePairFrontShortVEnv(old.parent.FrontShortVEnv):
    def _load_model(self):
        super()._load_model()
        contact=self.model.root.find('contact')
        if contact is None:contact=ET.SubElement(self.model.root,'contact')
        if list(contact.findall('pair')):raise RuntimeError('Own source had unexpected explicit contact pairs')
        for row in PARAMS['rows']:
            for name in (row['can'],row['rail']):
                if self.model.worldbody.find(f".//geom[@name='{name}']") is None:raise RuntimeError('Exact source-support geom absent: '+name)
            ET.SubElement(contact,'pair',name='r302_source_pair_'+row['rail'].rsplit('_',1)[-1],
                geom1=row['can'],geom2=row['rail'],condim=str(row['dim']),
                solref=text_vector(row['new_solref']),solreffriction=text_vector(row['old_solreffriction']),
                solimp=text_vector(row['solimp']),friction=text_vector(row['friction']),
                margin=format(row['margin'],'.17g'),gap=format(row['gap'],'.17g'))

_RAISED=FunctionType(old._RAISED_INIT.__code__,dict(old._RAISED_INIT.__globals__,H=H),closure=old._RAISED_INIT.__closure__)
_ORIGINAL_FRONT=FunctionType(old._FRONT_INIT.__code__,dict(old._FRONT_INIT.__globals__,H=H,_RAISED_INIT=_RAISED,
                           FrontShortVEnv=SourcePairFrontShortVEnv),closure=old._FRONT_INIT.__closure__)

def selected_front_init(self,*args,**kwargs):
    _ORIGINAL_FRONT(self,*args,**kwargs)
    m=self.backend.sim.model._model
    previous=mujoco.MjModel.from_binary_path(str(old.H/'startup_seed0/compiled_model.mjb'))
    changed=[];same=[]
    for n in dir(m):
        try:a,b=getattr(previous,n),getattr(m,n)
        except Exception:continue
        if not isinstance(a,np.ndarray) or not isinstance(b,np.ndarray):continue
        (same if a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes() else changed).append(n)
    allowed=lambda n:n.startswith('pair_') or n=='name_pairadr'
    bad=[n for n in changed if not allowed(n)]
    details=[]
    for i in range(m.npair):
        names=[m.geom(int(x)).name for x in (m.pair_geom1[i],m.pair_geom2[i])]
        row=next((r for r in PARAMS['rows'] if set(names)=={r['can'],r['rail']}),None)
        if row is None:raise RuntimeError('Unexpected compiled source-support pair')
        checks={'condim':int(m.pair_dim[i])==row['dim'],
            'solref':np.array_equal(m.pair_solref[i],row['new_solref']),
            'solreffriction':np.array_equal(m.pair_solreffriction[i],row['old_solreffriction']),
            'solimp':np.array_equal(m.pair_solimp[i],row['solimp']),
            'friction':np.array_equal(m.pair_friction[i],row['friction']),
            'margin':float(m.pair_margin[i])==row['margin'],'gap':float(m.pair_gap[i])==row['gap']}
        details.append({'pair_name':m.pair(i).name,'geoms':names,'checks':checks,'effective_solref':m.pair_solref[i].tolist()})
    passed=bool(m.npair==2 and not bad and all(all(x['checks'].values()) for x in details)
                and m.opt.timestep==PARAMS['dt_s'])
    (H/'compiled_pair_guard.json').write_text(json.dumps({'passed':passed,'same_model_array_count':len(same),
      'changed_model_arrays':changed,'unexpected_nonpair_changed_arrays':bad,'compiled_pairs':details,
      'dt_s':m.opt.timestep,'can_geom_solref_unchanged':m.geom_solref[m.geom('steel_thermos_1_can_body_collision').id].tolist(),
      'before_any_startup_controls':True},indent=2)+'\n')
    if not passed:raise RuntimeError('First compiled source-support pair/nonvisual guard failed')

_INIT=FunctionType(old.Fixed80FrontSlotCanScene.__init__.__code__,
                   dict(old.Fixed80FrontSlotCanScene.__init__.__globals__,H=H,_FRONT_INIT=selected_front_init,
                        sample=FunctionType(old.sample.__code__,dict(old.sample.__globals__,H=H)),verify=verify),
                   closure=old.Fixed80FrontSlotCanScene.__init__.__closure__)
class SourcePairFixed80CanScene(old.Fixed80FrontSlotCanScene):
    layout_version='R302-same-F-shortV-fixed80-two-source-support-pairs-solref20ms-v1'
    __init__=_INIT
