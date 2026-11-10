"""One bounded P2-02 real qualification on preregistered seeds0..9."""
import argparse,hashlib,json,time
from pathlib import Path
from research.v4.candidates.e3_02_teapot_teacup_20260926 import qualify as common
from .scene import HERE,ROOT
from . import scripts
def hashes():
 s=common.hashes()
 names=['scene_blade_side_v3.py','evaluator.py','scripted_blade_side_v3.py']
 paths=[ROOT/'research/v4/candidates/p2_v42_t3_20260925'/n for n in names]
 paths+=[ROOT/'research/v3/tasks/p2/evaluator.py',ROOT/'research/v3/tasks/p2/scene.py',ROOT/'research/v4/accelerated/scoring.py',ROOT/'research/upstream/polyhaven/wood_table_diff_1k.png']
 paths+=[*HERE.glob('*.py'),HERE/'protocol.json']
 paths+=[p for p in (ROOT/'research/v4/candidates/p1_03_scissors_case_20260926/assets').glob('*')if p.is_file()]
 for p in paths:s[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 f=HERE/'source_freeze.json'
 if f.exists()and s!=json.loads(f.read_text()):raise ValueError('P202 source drift')
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,choices=(3,10),default=3);a=p.parse_args();out=HERE/f'qualification_seed0to{a.seeds-1}.json'
 if out.exists():raise ValueError('keep prior result')
 s=hashes();start=time.monotonic();rows=[];error=None
 if a.seeds==10:
  old=json.loads((HERE/'qualification_seed0to2.json').read_text());assert old['passed']and old['source_sha256']==s;rows=old['rows']
 try:
  for seed in range(len(rows)//3,a.seeds):
   for mode in scripts.EXPECTED:
    r=scripts.probe(mode,seed);rows.append(r);print(json.dumps({k:r[k]for k in('seed','mode','accepted','error','steps')}),flush=True)
    if not r['accepted']:raise RuntimeError('firstphysicalfailure;stop')
 except Exception as e:error=repr(e)
 counts={m:sum(r['accepted']and r['mode']==m for r in rows)for m in scripts.EXPECTED}
 report=dict(instance='P2-02',kind='scripted CPU engineering qualification;not method or native result',rows=rows,counts=counts,error=error,new_elapsed_s=time.monotonic()-start,source_sha256=s,source_drift=hashes()!=s,passed=error is None and all(n==a.seeds for n in counts.values()))
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in('passed','counts','error','new_elapsed_s')}),flush=True);return 0 if report['passed']else 2
if __name__=='__main__':raise SystemExit(main())
