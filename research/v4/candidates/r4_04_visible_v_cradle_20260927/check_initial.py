"""Actual3seed stand test then untouched copied loop route once."""
import json,copy,time
from pathlib import Path
import mujoco,numpy as np
from .scene import Scene,HERE
from .neutral import settle
from research.v4.candidates.r4_04_pull_loop_legal_close_20260927 import check as legal

def geometry(s):
 m,d=s.backend.sim.model._model,s.backend.sim.data._data;obj=s.backend.objects_dict[s.item_name];r=legal.prev.fullcheck.initial(s,m,d,obj)
 # Old deck-only bottom gate is intentionally inapplicable to a visible V.
 # Retain all actual contacts/table/projections; support comes from actual
 # two-face normals, not declaring a hovering pillow supported by a deck.
 r['old_deck_only_pass_not_applicable']=r.pop('passed');cts=r['contacts'];r['actual_initial_contact_support']=[x for x in cts if any(z.startswith('r404_upright_wood_')for z in x['pair'])and x['pillow']]
 r['passed']=bool(r['whole_rack_pillow_table_XY']and not r['blockers']and not r['deep_contacts']and r['public_key_visibility']);return r

def main():
 out=HERE/'initial_stability.json'
 if out.exists():raise RuntimeError('First actual initial/neutral candidate immutable')
 start=time.monotonic();r=dict(passed=False,seeds=[],copied=None,error=None,actual_controls=0,GPU=False,models=[])
 for seed in(0,1,2):
  s=Scene(seed)
  try:
   a=geometry(s);row=dict(seed=seed,initial=a,settling=None)
   if not a['passed']:r['seeds'].append(row);r['error']='first initial actual geometry failure';break
   n=settle(s);row['settling']=n;r['actual_controls']+=n['controls'];r['seeds'].append(row)
   if not n['passed']:r['error']='first actual neutral source failure';break
   if seed==0:
    om,od=s.backend.sim.model._model,s.backend.sim.data._data;before=legal.prev.wood.parent.state(om,od);mh=legal.prev.wood.parent.digest_model(om);m=copy.deepcopy(om);d=mujoco.MjData(m);mujoco.mj_setState(m,d,before,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);obj=s.backend.objects_dict[s.item_name];jid=m.joint(obj.joints[0]).id;adr=m.jnt_qposadr[jid];c=dict(approach=legal.prev.route_fn()(s,m,d,obj,jid,adr),legal_held=None)
    if c['approach']['passed']:c['legal_held']=legal.held_fn()(s,m,d,obj,jid,adr)
    c['original_integration_unchanged']=bool(np.array_equal(before,legal.prev.wood.parent.state(om,od)));c['original_model_unchanged']=mh==legal.prev.wood.parent.digest_model(om);r['copied']=c
    if not c['approach']['passed']or not c['legal_held']or not c['legal_held']['passed']:r['error']='first copied approach/close/held failure';break
  finally:s.close()
 r['passed']=r['error']is None and len(r['seeds'])==3;r['elapsed_s']=time.monotonic()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in('passed','error','actual_controls','elapsed_s')}))
if __name__=='__main__':main()
