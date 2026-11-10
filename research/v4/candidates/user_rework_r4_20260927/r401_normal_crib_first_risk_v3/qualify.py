"""Unique predeclared own cases; first real failure stops, no reruns."""
import argparse,hashlib,json,signal,time
from .scripts import run_case,HERE,ROOT
def hashes():
 pins=json.loads((HERE/'source_freeze.json').read_text())
 for n,h in pins.items():
  if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Own R401 source drift '+n)
 return pins
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',choices=['seed0','seeds1_2','seeds3_9'],required=True);a=p.parse_args();pins=hashes();out=HERE/a.batch
 if out.exists():raise RuntimeError('Unique actual output required; no retry')
 if a.batch!='seed0':
  required=['seed0']if a.batch=='seeds1_2'else['seed0','seeds1_2']
  for name in required:
   if not json.loads((HERE/name/'summary.json').read_text())['passed']:raise RuntimeError('Prior predeclared cases failed')
 out.mkdir();rows=[];start=time.monotonic();old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('Own90s script cap')))
 try:
  for seed in [0]if a.batch=='seed0'else range(1,3)if a.batch=='seeds1_2'else range(3,10):
   for case in('correct','naive','capability_failure'):
    signal.alarm(90);r=run_case(seed,case);signal.alarm(0);r.update(source_sha256=pins,source_drift=pins!=hashes());r['passed']&=not r['source_drift'];(out/f'{case}_seed{seed}.json').write_text(json.dumps(r,indent=2)+'\n')
    row=dict(seed=seed,case=case,passed=r['passed'],control_steps=r['control_steps'],label=r['evaluator']['label'],held=r.get('held_after_lift',False),lift_m=r.get('lift_m'),error=r.get('error'),unexpected_contacts=len(r.get('unexpected_risk_contacts',[])))
    rows.append(row);print(json.dumps(row),flush=True)
    if not row['passed']:return 2
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old);expected={'seed0':3,'seeds1_2':6,'seeds3_9':21}[a.batch];summary=dict(batch=a.batch,rows=rows,passed=len(rows)==expected and all(x['passed']for x in rows),elapsed_s=time.monotonic()-start,source_sha256=pins,source_drift=pins!=hashes(),native_pending=True,old_or_other_instance_qualification_inherited=False,models_called=[],GPU=False);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
