"""Finite preregistered0/1/2 then3..9, real scripts only; no native acceptance."""
import argparse,hashlib,json,signal,time,xml.etree.ElementTree as ET
from pathlib import Path
from .scene import HERE,ROOT
from .scripts import run_case
def hashes():
 files=set(HERE.glob('*.py'))|{HERE/'config_v1.json',HERE/'asset_meta.json'}|set((HERE/'assets').glob('*'))
 for n in ('research/v4/accelerated/tabletop.py','research/v4/accelerated/grasp_assist.py','research/v4/accelerated/scoring.py','research/v4/accelerated/observation.py','research/v4/tasks/r4/template.py','research/v4/tasks/r4/evaluator.py','research/v4/tasks/r4/assets.json','research/v4/tasks/r4/crib.xml','research/v4/configs/instance_registry.json','research/v4/candidates/r3_v42_tabletop_20260925/scene.py','research/v4/candidates/r3_original_moka_integration/candidate.py','research/v4/candidates/r2_03_t1_derived_20260926/selected_pose_v3.json','research/v4/bridge/run_b0.py','research/upstream/polyhaven/wood_table_diff_1k.png'):
  files.add(ROOT/n)
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='qualification_v1');p.add_argument('--seeds',default='0,1,2');a=p.parse_args();seeds=list(map(int,a.seeds.split(',')));out=HERE/a.output
 if out.exists():raise ValueError('fresh output required')
 out.mkdir();before=hashes();started=time.monotonic();rows=[]
 def alarm(*_):raise TimeoutError('R4-06 CPU batch1800s')
 old=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,alarm);signal.alarm(1800)
 try:
  for seed in seeds:
   for case in ('correct','naive','capability_failure'):
    r=run_case(seed,case);after=hashes();r.update(source_sha256=before,source_drift=[n for n,h in before.items() if after.get(n)!=h]);r['passed']&=before==after
    (out/f'{case}_seed{seed}.json').write_text(json.dumps(r,indent=2)+'\n');row={k:r[k] for k in ('case','seed','passed','control_steps','elapsed_s','error')};row['label']=r['evaluator']['label'];rows.append(row);print(json.dumps(row),flush=True)
    if not r['passed']:return 2
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old);after=hashes()
  report={'kind':'R4-06 CPU script qualification, not native/method result','rows':rows,'pass_counts':{c:sum(r['passed'] for r in rows if r['case']==c) for c in ('correct','naive','capability_failure')},'predeclared_seeds':[0,1,2],'followup_seeds':list(range(3,10)),'passed':len(rows)==len(seeds)*3 and all(r['passed'] for r in rows) and before==after,'source_sha256':before,'source_drift':[n for n,h in before.items() if after.get(n)!=h],'elapsed_s':time.monotonic()-started}
  (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('passed','pass_counts','source_drift','elapsed_s')}),flush=True)
 return 0 if report['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
