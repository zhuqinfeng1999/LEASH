"""Finite preregistered0/1/2 then3..9, real scripts only; no native acceptance."""
import argparse,hashlib,json,signal,time,xml.etree.ElementTree as ET
from pathlib import Path
from .scene import HERE,ROOT
from .scripts import run_case
def hashes():
 files=set(HERE.glob('*.py'))|{HERE/'config_v1.json'}|set((HERE/'assets').glob('*'))
 for n in ('research/v4/accelerated/tabletop.py','research/v4/accelerated/grasp_assist.py','research/v4/accelerated/scoring.py','research/v4/accelerated/observation.py','research/v4/tasks/r3/template.py','research/v4/tasks/r3/evaluator.py','research/v4/tasks/r3/assets.json','research/v4/configs/instance_registry.json','research/v4/candidates/r3_v42_tabletop_20260925/scene.py','research/v4/candidates/r3_v42_tabletop_20260925/microwave_fullsize_180.xml','research/v4/candidates/r3_v42_visibility_20260926/scene_lower_aim.py','research/v4/candidates/r3_v42_visibility_20260926/cpu_lower_aim_search_seed0.json','research/v4/candidates/r3_original_moka_integration/candidate.py','research/v4/bridge/run_b0.py'):
  files.add(ROOT/n)
 files.update(p for p in (ROOT/'upstream/LIBERO/libero/libero/assets/articulated_objects/meshes/microwave').rglob('*') if p.is_file())
 files.update(p for p in (ROOT/'upstream/LIBERO/libero/libero/assets/articulated_objects/textures').glob('metal1.png'))
 # The inherited oven XML references the actual data/libero copies. Pin these
 # exact consumed files as well as upstream provenance, without duplicating
 # the assets or modifying the accepted device XML.
 oven=ROOT/'research/v4/candidates/r3_v42_tabletop_20260925/microwave_fullsize_180.xml'
 for node in ET.parse(oven).getroot().find('asset'):
  if node.get('file'):
   dependency=Path(node.get('file'))
   if not dependency.is_absolute():dependency=oven.parent/dependency
   if not dependency.is_file():raise FileNotFoundError(dependency)
   files.add(dependency)
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='qualification_v1');a=p.parse_args();out=HERE/a.output
 if out.exists():raise ValueError('fresh output required')
 out.mkdir();before=hashes();started=time.monotonic();rows=[]
 def alarm(*_):raise TimeoutError('R3-03 CPU batch1800s')
 old=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,alarm);signal.alarm(1800)
 try:
  for seed in range(10):
   for case in ('correct','naive','capability_failure'):
    r=run_case(seed,case);after=hashes();r.update(source_sha256=before,source_drift=[n for n,h in before.items() if after.get(n)!=h]);r['passed']&=before==after
    (out/f'{case}_seed{seed}.json').write_text(json.dumps(r,indent=2)+'\n');row={k:r[k] for k in ('case','seed','passed','control_steps','elapsed_s','error')};row['label']=r['evaluator']['label'];rows.append(row);print(json.dumps(row),flush=True)
    if not r['passed']:return 2
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old);after=hashes()
  report={'kind':'R3-03 CPU script qualification, not native/method result','rows':rows,'pass_counts':{c:sum(r['passed'] for r in rows if r['case']==c) for c in ('correct','naive','capability_failure')},'predeclared_seeds':[0,1,2],'followup_seeds':list(range(3,10)),'passed':len(rows)==30 and all(r['passed'] for r in rows) and before==after,'source_sha256':before,'source_drift':[n for n,h in before.items() if after.get(n)!=h],'elapsed_s':time.monotonic()-started}
  (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('passed','pass_counts','source_drift','elapsed_s')}),flush=True)
 return 0 if report['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
