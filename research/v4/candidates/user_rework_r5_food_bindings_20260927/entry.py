"""Fresh finite R5 meal appearance previews/HD through original own exporters."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from .binding import HERE,ROOT,ROWS,scene_class,FOOD
DONORS={
 'R5-01':'scene_review_r5_01_20260927.export',
 'R5-02':'scene_review_r5_02_03_20260927.export',
 'R5-03':'scene_review_r5_02_03_20260927.export',
 'R5-04':'scene_review_r5_04_05_06_20260927.export',
 'R5-05':'scene_review_r5_04_05_06_20260927.export',
 'R5-06':'scene_review_r5_04_05_06_20260927.export',
}
def hashes(require_freeze=True):
 report=json.loads((HERE/'cpu_constructor_contract.json').read_text())
 if not report['accepted']or len(report['rows'])!=6 or any(not r['exact']for r in report['rows']):raise ValueError('Six actual physical identity witnesses required')
 pins={}
 for instance in ROWS:
  sidecar=ROOT/f'research/v4/review_videos/{instance}.json'
  for key in json.loads(sidecar.read_text())['source_sha256']:
   p=Path(key)if key.startswith('/')else ROOT/key
   if not p.is_file():raise ValueError('Missing own historical source reference:'+key)
   pins[key]=hashlib.sha256(p.read_bytes()).hexdigest()
  pins[str(sidecar.relative_to(ROOT))]=hashlib.sha256(sidecar.read_bytes()).hexdigest()
 for p in [*HERE.glob('*.py'),HERE/'cpu_constructor_contract.json',*FOOD.parent.glob('*.obj'),FOOD,FOOD.parent/'LICENSE.txt',FOOD.parent/'build_assets.py']:
  pins[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 pins=dict(sorted(pins.items()))
 if require_freeze:
  f=HERE/'source_freeze.json'
  if not f.is_file()or json.loads(f.read_text())!=pins:raise ValueError('Own meal binding/source admission mismatch')
 return pins
def selected(instance):
 if instance not in ROWS:raise ValueError('Exact own R5 allowlist required')
 donor=importlib.import_module('research.v4.candidates.'+DONORS[instance]);original=donor.Configuration;own=scene_class(instance)
 class OwnConfiguration(original):
  def __init__(self,selected=instance):
   if selected!=instance:raise ValueError('Scoped own instance mismatch')
   super().__init__(selected);self.Scene=own
 return donor,OwnConfiguration
@contextmanager
def scope(instance):
 donor,configuration=selected(instance)
 with patch.multiple(donor,HERE=HERE,hashes=hashes,Configuration=configuration):yield donor
def main():
 p=argparse.ArgumentParser();p.add_argument('--instance',choices=list(ROWS),required=True);p.add_argument('--egl-device',type=int,default=0);p.add_argument('--physical-gpu',type=int,default=0);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--preflight-only',action='store_true');g.add_argument('--preview-only',action='store_true');g.add_argument('--capture',action='store_true');a=p.parse_args();pins=hashes()
 if a.preflight_only:
  donor,cfg=selected(a.instance);s=cfg(a.instance);print(json.dumps(dict(instance=a.instance,source_count=len(pins),selected_class=s.Scene.__name__,fresh_preview=not(HERE/a.instance/'preview').exists(),fresh_capture=not(HERE/a.instance/'capture').exists(),Scene_constructed=False,GPU=False,models=[])));return
 mode='--preview-only'if a.preview_only else'--capture';argv=[sys.argv[0],'--egl-device',str(a.egl_device),'--physical-gpu',str(a.physical_gpu),mode]
 if a.instance!='R5-01':argv.extend(['--instances',a.instance])
 try:
  with scope(a.instance)as donor,patch.object(sys,'argv',argv):donor.main()
 finally:
  for f in [HERE/a.instance/'preview/report.json',HERE/a.instance/'capture/report.json',HERE/a.instance/(a.instance+'.candidate.json')]:
   if f.exists():
    r=json.loads(f.read_text());r.update(source_code_version='R5-own-authored-meal-visual-only-v1',only_lunchbox_visuals_changed=True,old_physics_reused_by_exact_constructor_proof=True,constructor_identity_evidence=str((HERE/'cpu_constructor_contract.json').relative_to(ROOT)),appearance_root_review_pending=True,models_called=[],canonical_overwritten=False);f.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
