"""Finite prescribed seed qualification, first actual failure ends branch."""
import argparse,hashlib,json,signal,time
from pathlib import Path
from . import scripts
from research.v4.candidates.r4_01_raised3mm_first_risk_20260926 import preflight as parent
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def hashes():
 pins=parent.hashes()
 for p in [*HERE.glob('*.py'),HERE/'protocol.json',HERE/'HYPOTHESIS.md',ROOT/'research/v4/candidates/r4_03_neck_layout_readonly_20260927/check_geometry.py',ROOT/'research/v4/candidates/r4_04_side_pinch_clearance_v2_20260927/derive.py']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 frozen=HERE/'source_freeze.json'
 if frozen.exists()and json.loads(frozen.read_text())!=pins:raise RuntimeError('new offline script source drift')
 return pins
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['first3','followup7'],required=True);a=p.parse_args();out=HERE/(a.stage+'.json')
 if out.exists():raise RuntimeError('retain actual evidence; no retry')
 if a.stage=='followup7'and not json.loads((HERE/'first3.json').read_text())['passed']:raise RuntimeError('three seed qualification required')
 pins=hashes();rows=[];start=time.monotonic()
 def timeout(*_):raise TimeoutError('actual script90s bound')
 old=signal.signal(signal.SIGALRM,timeout)
 try:
  for seed in(range(3)if a.stage=='first3'else range(3,10)):
   for case in('correct','naive','capability_failure'):
    signal.alarm(90);r=scripts.run_case(seed,case);signal.alarm(0);r['source_sha256']=pins;r['source_drift']=pins!=hashes();r['passed']&=not r['source_drift'];(HERE/f'{case}_seed{seed}.json').write_text(json.dumps(r,indent=2)+'\n');row=dict(seed=seed,case=case,passed=r['passed'],actual_controls=r['control_steps'],label=r['evaluator']['label'],external_contacts=len(r['unexpected_risk_contacts']));rows.append(row);print(json.dumps(row),flush=True)
    if not row['passed']:return 2
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old);report=dict(stage=a.stage,rows=rows,passed=len(rows)==(9 if a.stage=='first3' else 21)and all(x['passed']for x in rows),elapsed_s=time.monotonic()-start,source_sha256=pins,source_drift=pins!=hashes(),models=[],GPU_calls=0,native_pending=True,old_motion_qualifications_not_inherited=True);out.write_text(json.dumps(report,indent=2)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
