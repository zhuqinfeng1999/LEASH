"""Bounded same-source first3->ten, preserve all failures and original labels."""
import argparse,hashlib,json,time,signal
from .scripts import HERE,ROOT,run_case
from research.v4.candidates.r3_06_original_aluminum_tray_20260926.qualify_cpu import hashes as oldhashes
def hashes():
 out=oldhashes()
 for p in [*HERE.glob('*.py'),HERE/'protocol.json']:out[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 f=HERE/'source_freeze.json'
 if f.exists()and out!=json.loads(f.read_text()):raise ValueError('R306 new path source drift')
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,choices=[3,10],default=3);a=p.parse_args();out=HERE/f'qualification_{a.seeds}'
 if out.exists():raise ValueError('fresh outputs required')
 out.mkdir();pins=hashes();rows=[];start=time.monotonic();error=None
 if a.seeds==10:
  old=json.loads((HERE/'qualification_3/summary.json').read_text());assert old['passed']and old['source_sha256']==pins;rows=old['rows']
 def alarm(*_):raise TimeoutError('new R306 CPU batch900s')
 oldhandler=signal.signal(signal.SIGALRM,alarm);signal.alarm(900)
 try:
  for seed in range(len(rows)//3,a.seeds):
   for case in ['correct','naive','capability_failure']:
    r=run_case(seed,case);r['source_drift']=hashes()!=pins;r['passed']&=not r['source_drift'];(out/f'{case}_seed{seed}.json').write_text(json.dumps(r,indent=2)+'\n');row={k:r[k]for k in ['case','seed','passed','elapsed_s','control_steps','error']};row['label']=r['evaluator']['label'];rows.append(row);print(json.dumps(row),flush=True)
    if not r['passed']:raise RuntimeError('first physical failure; stop branch')
 except Exception as e:error=repr(e)
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,oldhandler);counts={c:sum(x['passed']and x['case']==c for x in rows)for c in ['correct','naive','capability_failure']};report=dict(instance='R3-06',kind='CPU scripted engineering; no native/model',source_sha256=pins,source_drift=hashes()!=pins,rows=rows,counts=counts,error=error,new_elapsed_s=time.monotonic()-start,passed=error is None and len(rows)==3*a.seeds);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['passed','counts','error','new_elapsed_s']}),flush=True)
 return 0 if report['passed']else 2
if __name__=='__main__':raise SystemExit(main())
