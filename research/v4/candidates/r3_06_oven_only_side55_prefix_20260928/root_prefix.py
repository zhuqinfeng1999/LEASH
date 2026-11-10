"""Actual accepted SidePath + original55turn, scoped existing root logger."""
import ast,hashlib,inspect,json,sys
from pathlib import Path
from unittest.mock import patch
from itertools import product
import numpy as np
from research.v4.candidates.r3_05_oven_only_prefix_20260928 import root_prefix as recorder
from research.v4.candidates.r3_02_oven_only_prefix_20260928.root_prefix import contact_rows,ORIGINAL_INITIAL_NONDEVICE
from research.v4.candidates.r3_06_side55_entry_20260926 import scripts as own
from research.v4.candidates.r3_06_mouth_frame_20260926 import scripts as frame
from research.v4.candidates.r3_06_mouth_path_20260926 import scripts as transport
from research.v4.candidates.r3_03_t1_derived_20260926 import scripts as shared
from .scene import OvenOnlyTrayScene,OriginalRecordedScene
H=Path(__file__).resolve().parent
ORIGINAL_RISK_SNAPSHOT=recorder.snapshot
INITIAL_METADATA=[]

def initial_nondevice(scene):
 result=ORIGINAL_INITIAL_NONDEVICE(scene);sim=scene.backend.sim;m,d=sim.model._model,sim.data._data;item=scene.backend.objects_dict[scene.item_name];oven=scene.backend.objects_dict[scene.oven_name]
 INITIAL_METADATA.append({'class':type(scene).__module__+'.'+type(scene).__name__,'actual_oven_qpos':np.asarray(sim.data.get_joint_qpos(oven.joints[0])).tolist(),'actual_item_qpos':np.asarray(sim.data.get_joint_qpos(item.joints[0])).tolist(),'actual_fixture_eq_data':m.eq_data[m.equality('r3_microwave_fixture').id].tolist(),'sequence':scene.sequence,'time_s':float(d.time),'offline_only':True})
 return result

def snapshot(scene):
 row=ORIGINAL_RISK_SNAPSHOT(scene);m,d=scene.backend.sim.model._model,scene.backend.sim.data._data;c,R=transport._mouth(scene);witness=[]
 for g in range(m.ngeom):
  name=m.geom(g).name or ''
  if not(name.startswith('steel_thermos_1_')and m.geom_type[g]==6 and(m.geom_contype[g]or m.geom_conaffinity[g])):continue
  points=np.array([m.geom_size[g]*s for s in product((-1,1),repeat=3)]);local=(d.geom_xpos[g]+points@d.geom_xmat[g].reshape(3,3).T-c)@R;inside=np.all((local>transport.OPEN_LO+.001)&(local<transport.OPEN_HI-.001),axis=1)
  if np.any(inside):witness.append({'actual_metal_box_geom':name,'actual_vertices_inside':local[inside].tolist()})
 # Exact own accepted metal-vertex geometry is diagnostic ONLY. Original strict
 # center/door risk fields never changed or replaced by this extra witness.
 row['own_real_metal_vertices']={'offline_only':True,'found':bool(witness),'witnesses':witness,'whole_tray_contained_claim':False};return row

def compiled_prefix(Scene=OvenOnlyTrayScene):
 # Reuse accepted actual AST selection instead of stopped original tray script.
 original=ast.parse(Path(shared.__file__).read_text());accepted=ast.fix_missing_locations(own.SidePath().visit(original))
 fn=next(n for n in accepted.body if isinstance(n,ast.FunctionDef)and n.name=='run_case');before=ast.dump(fn,include_attributes=False);original_src=ast.unparse(fn)
 count=0;prefix=None
 class AfterTurn(ast.NodeTransformer):
  def visit_Assign(self,node):
   nonlocal count,prefix
   if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='_prepare_path':
    count+=1
    return [*ast.parse("details.update(prefix_only=True,attached=attached,lift_m=lift,held_after_lift=bool(attached and lift>=.020),passed=bool(attached and lift>=.020),terminal_status='side55_prefix_diagnostic_complete',evaluator=evaluator.finish('done'))\nreturn details").body,node]
   return self.generic_visit(node)
 selected=ast.fix_missing_locations(AfterTurn().visit(fn));src=ast.unparse(selected)
 seam='            staging_high, staging_low, insertion_target = _prepare_path(scene, details)';orig_at=original_src.index(seam)
 addition='            details.update(prefix_only=True';new_at=src.index(addition)
 if count!=1 or src[:new_at]!=original_src[:orig_at]:raise ValueError('Accepted actual own prefix/55turn source differs before terminal injection')
 if '_turn_side(scene, command, evaluator, details)'not in src[:new_at]:raise ValueError('Own original actual55turn missing')
 ns=dict(own.namespace);ns.update(R3TabletopFacade=Scene)
 exec(compile(ast.Module(body=[selected],type_ignores=[]),str(H/'root_prefix.py')+'::accepted-own55prefix-stop-before-path','exec'),ns)
 return ns['run_case'],src,hashlib.sha256(src[:new_at].encode()).hexdigest()

def main():
 # The donor's finite allocator/actual recorder/first-failure semantics are
 # unchanged. Restore all module globals and own original observer lists.
 pools=[own.records,frame.records,transport.records];sizes=[len(x)for x in pools];INITIAL_METADATA.clear()
 try:
  with patch.multiple(recorder,H=H,T1R3ForkBowlScene=OriginalRecordedScene,OvenOnlyPrefixScene=OvenOnlyTrayScene,compiled_prefix=compiled_prefix,contact_rows=contact_rows,initial_nondevice=initial_nondevice,snapshot=snapshot):return recorder.main()
 finally:
  for pool,n in zip(pools,sizes):del pool[n:]
  if '--output'in sys.argv:
   out=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
   if out.is_dir():(out/'actual_initial_own_source_device_metadata.json').write_text(json.dumps(INITIAL_METADATA,indent=2)+'\n')
if __name__=='__main__':main()
