"""Full physical baby↔crib/table scan incl static same-body geometry."""
import json,time
import mujoco,numpy as np
from .candidate import HERE,scene_factory,parent
from research.v4.candidates.r4_06_flat_underbase_v2_20260926.project_cpu import screen

def scan(seed):
 s=scene_factory()(seed)
 try:
  m,d=s.backend.sim.model._model,s.backend.sim.data._data;baby=[g for g in range(m.ngeom)if '_doll_'in(m.geom(g).name or'')and(m.geom_contype[g]or m.geom_conaffinity[g])];crib=[g for g in range(m.ngeom)if(m.geom(g).name or'').startswith('crib_1_')and '_doll_'not in(m.geom(g).name or'')and(m.geom_contype[g]or m.geom_conaffinity[g])];table=[g for g in range(m.ngeom)if(m.geom(g).name or'').startswith('table')and(m.geom_contype[g]or m.geom_conaffinity[g])];pairs=[]
  for a in baby:
   for b in crib+table:pairs.append(dict(baby=m.geom(a).name,other=m.geom(b).name,distance_m=float(mujoco.mj_geomDistance(m,d,a,b,.03,None))))
  bad=[p for p in pairs if p['distance_m']<0.];project=screen(s.backend.sim);contacts=[]
  for c in d.contact[:d.ncon]:
   names=[m.geom(int(i)).name or''for i in(c.geom1,c.geom2)]
   if any('_doll_'in n for n in names)or any(n.startswith(('robot0_','gripper0_'))for n in names):contacts.append(dict(geoms=names,distance_m=float(c.dist)))
  # Skin/collider world pose matching plus original body inertial proof.
  match=[]
  for gid in baby:
   other=m.geom((m.geom(gid).name or'').removesuffix('_risk_collision')).id
   match.append(dict(collider=m.geom(gid).name,match=bool(np.array_equal(d.geom_xpos[gid],d.geom_xpos[other])and np.array_equal(d.geom_xmat[gid],d.geom_xmat[other])and np.array_equal(m.geom_size[gid],m.geom_size[other])and m.geom_type[gid]==m.geom_type[other])))
  return dict(seed=seed,passed=len(baby)==19 and not bad and not contacts and project['screen_pass']and all(p['match']for p in match),actual_same_body_geometric_pairs=pairs,penetrations=bad,initial_risk_contacts=contacts,visual_collision_match=match,public_geometry=project,crib_root_mass_kg=float(m.body_mass[m.body(s.backend.objects_dict[s.target_name].root_body).id]))
 finally:s.close()
def main():
 out=HERE/'geometry10.json'
 if out.exists():raise RuntimeError('Preserve first qualification')
 start=time.monotonic();rows=[]
 try:
  for seed in range(10):
   r=scan(seed);rows.append(r);print(json.dumps(dict(seed=seed,passed=r['passed'],penetrations=r['penetrations'])),flush=True)
   if not r['passed']:return 2
 finally:(out).write_text(json.dumps(dict(kind='CPU initial physical/visual geometry, no native acceptance',rows=rows,passed=len(rows)==10 and all(r['passed']for r in rows),elapsed_s=time.monotonic()-start),indent=2)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
