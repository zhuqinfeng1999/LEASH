"""Run only the pre-registered six saved packets per visual API, once each."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
from .backend import DecisionBackend,JevBackend,parity,build_request

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'research/v4/results/decision_gate_dev_20261019'

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=('openai_decisions','jev','both'),default='both');args=parser.parse_args()
 plan=json.loads((OUT/'review_plan.json').read_text());resultfile=OUT/'results.json'
 results=json.loads(resultfile.read_text())if resultfile.exists()else[]
 for name,cls in (('openai_decisions',DecisionBackend),('jev',JevBackend)):
  if args.backend not in ('both',name):continue
  b=cls(OUT/name)
  for case in plan['cases']:
   key=(name,case['id'])
   if any((r['backend'],r['case_id'])==key for r in results):continue
   print('START',name,case['id'],flush=True)
   row=dict(backend=name,case_id=case['id'],gate=case['gate'],run_id=case['run_id'],instance=case['instance'],started_utc=datetime.now(timezone.utc).isoformat())
   try:
    _,packet=build_request(name,case['gate'],case['evidence'])
    if parity(packet)!=case['input_parity']:raise RuntimeError('Registered input bytes changed')
    row.update(result=b.score(case['gate'],case['evidence']),status='completed')
   except Exception as exc:row.update(status='failed',error=str(exc))
   results.append(row);temp=resultfile.with_suffix('.tmp');temp.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(resultfile)
   print('END',name,case['id'],row['status'],row.get('result',{}).get('p_safe'),row.get('error',''),flush=True)
 b.close()
if __name__=='__main__':main()
