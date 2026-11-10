"""Thin R2-03 binding over the already native R2-02 method entry."""
import json,hashlib
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.r2_02_natural_method_20260926 import run_method as base
from research.v4.candidates.r2_03_t1_derived_20260926.scene import T1R2TealightScene
from research.v4.candidates.r2_02_t1_derived_20260926.offline_evaluator import Evaluator
from .public_boundary import TealightReviewerWorker,TealightWorkspaceReader,TealightMotionReviewer,bind_task
HERE=Path(__file__).resolve().parent;ROOT=base.ROOT;INSTANCE='R2-03'
PROTOCOL='v4.2-dev-r2-03-assisted-640x480-48x144-80turn-wall1800-t1-public-v1'
CARRIER=HERE.parent/'r2_03_t1_derived_20260926'
BASE_HASHES=base.hashes
def hashes():
    source=BASE_HASHES();source.update(json.loads((CARRIER/'source-sha256.json').read_text()))
    source.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in HERE.glob('*.py')})
    return dict(sorted(source.items()))
def qualification(*,require_native=False):
    q=json.loads((CARRIER/'qualification_v3/summary.json').read_text())
    native=json.loads((CARRIER/'native_seed0_gpu1/report.json').read_text())
    accepted=json.loads((ROOT/'research/v4/results/r2_03_v42_instance_acceptance_20260926.json').read_text())
    assert q['passed'] and q['pass_counts']=={'correct':10,'naive':10,'ability_failure':10}
    for report,key in ((q,'source_sha256'),(native,'source_hashes')):
        drift=[n for n,h in report[key].items()if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
        if drift:raise RuntimeError('qualified R2-03 drift '+str(drift))
    if not native['passed_diagnostic']or native['source_drift']or not native['physical_gpu_placement']['observed']or not accepted['accepted']:raise RuntimeError('R2-03 native root acceptance missing')
    return {'script_counts':q['pass_counts'],'native_diagnostic_ready':True,'native_report':str(CARRIER/'native_seed0_gpu1/report.json'),'qualified_source_drift':[],'visual_promotion_owner':'root'}
def main(argv=None):
    with patch.multiple(base,Scene=T1R2TealightScene,Evaluator=Evaluator,INSTANCE=INSTANCE,INSTRUCTION=T1R2TealightScene.instruction,PROTOCOL=PROTOCOL,hashes=hashes,qualification=qualification,T1ReviewerWorker=TealightReviewerWorker,PublicWorkspaceReader=TealightWorkspaceReader,PublicMotionReviewer=TealightMotionReviewer,bind_task=bind_task):
        return base.main(argv)
if __name__=='__main__':raise SystemExit(main())
