"""Two accepted T1 configurations, one existing natural RPent/LEASH entry."""
import argparse,json,hashlib,sys
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.r2_02_natural_method_20260926 import run_method as base
from research.v4.candidates.r2_04_t1_derived_20260926.scene import T1R2CharcoalScene
from research.v4.candidates.r2_05_t1_derived_20260926.scene import T1R2DrawerScene
from research.v4.candidates.r2_02_t1_derived_20260926.offline_evaluator import Evaluator as CharcoalEvaluator
from research.v4.candidates.r2_05_t1_derived_20260926.offline_evaluator import Evaluator as DrawerEvaluator
from . import public_boundary as public
ROOT=base.ROOT;HERE=Path(__file__).resolve().parent;BASE_HASHES=base.hashes
CONFIG={
 'R2-04':(T1R2CharcoalScene,CharcoalEvaluator,'qualification_v3','native_seed0_gpu1',public.CharcoalReviewerWorker,public.PublicWorkspaceReader,public.PublicMotionReviewer,public.bind_charcoal),
 'R2-05':(T1R2DrawerScene,DrawerEvaluator,'qualification_v2','native_seed0_gpu0',public.DrawerReviewerWorker,public.DrawerWorkspaceReader,public.DrawerMotionReviewer,public.bind_drawer),
}
def protocol(instance):return f'v4.2-dev-{instance.lower()}-assisted-640x480-48x144-80turn-wall1800-t1-public-v1'
def carrier(instance):return HERE.parent/(instance.lower().replace('-','_')+'_t1_derived_20260926')
def hashes():
 source=BASE_HASHES()
 for instance in CONFIG:source.update(json.loads((carrier(instance)/'source-sha256.json').read_text()))
 source.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in HERE.glob('*.py')})
 freeze=HERE/'source_freeze.json'
 if freeze.exists() and source!=json.loads(freeze.read_text()):raise RuntimeError('new entry source drift')
 return dict(sorted(source.items()))
def qualification(instance,*,require_native=False):
 cfg=CONFIG[instance];p=carrier(instance);q=json.loads((p/cfg[2]/'summary.json').read_text());n=json.loads((p/cfg[3]/'report.json').read_text())
 accepted=json.loads((ROOT/f'research/v4/results/{instance.lower().replace("-","_")}_v42_instance_acceptance_20260926.json').read_text())
 assert q['passed'] and q['pass_counts']=={'correct':10,'naive':10,'ability_failure':10}
 for report,key in ((q,'source_sha256'),(n,'source_hashes')):
  drift=[k for k,h in report[key].items()if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=h]
  if drift:raise RuntimeError('qualified scene source drift '+str(drift))
 assert n['passed_diagnostic'] and not n['source_drift'] and n['physical_gpu_placement']['observed'] and accepted['accepted']
 return {'script_counts':q['pass_counts'],'native_diagnostic_ready':True,'native_report':str(p/cfg[3]/'report.json'),'qualified_source_drift':[],'visual_promotion_owner':'root'}
def main(argv=None):
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--instance',choices=CONFIG,required=True);a,rest=p.parse_known_args(sys.argv[1:]if argv is None else argv)
 scene,evaluator,_,_,worker,workspace,motion,bind=CONFIG[a.instance]
 with patch.multiple(base,Scene=scene,Evaluator=evaluator,INSTANCE=a.instance,INSTRUCTION=scene.instruction,PROTOCOL=protocol(a.instance),hashes=hashes,qualification=lambda **kw:qualification(a.instance,**kw),T1ReviewerWorker=worker,PublicWorkspaceReader=workspace,PublicMotionReviewer=motion,bind_task=bind):return base.main(rest)
if __name__=='__main__':raise SystemExit(main())
