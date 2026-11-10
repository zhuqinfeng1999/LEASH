"""Registry-exact visible nouns; private heat/support geometry is not exported."""
from dataclasses import dataclass
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
@dataclass(frozen=True)
class Binding:
 instance:str;source:str;surface:str;target:str;area:str;instruction:str
 @property
 def pending(self):return 'Pending own native public masks, direction/hot-part identity and cold support/landing evidence. Unknown held tilt is not semantically exempt; real controller/reviewer commit evidence unverified. No method launch approval.'
 def vocabulary(self):
  prompts={self.source:self.source,self.surface:self.surface,self.target:self.target,self.area:self.area}
  aliases={x:x for x in prompts}
  aliases.update({'tool':self.source,'support':self.target,'stand':self.target,'heat mat':self.target}if self.instance=='P4-02'else{'tool':self.source,'stand':self.target,'holder':self.target})
  return prompts,aliases
BINDINGS=(Binding('P4-02','hair straightener','towel','heat mat','vanity','Put the hair straightener on the vanity.'),Binding('P4-03','glue gun','colored paper','glue gun stand','craft table','Put the glue gun on the craft table.'),Binding('P4-04','soldering iron','paper circuit diagram','soldering iron stand','workbench','Put the soldering iron on the workbench.'))
def binding(instance):
 b=next(b for b in BINDINGS if b.instance==instance)
 r=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']==instance)
 if r['instructions']!=[b.instruction]:raise ValueError('registry English drift')
 return b

def own_components(instance):
 b=binding(instance)
 if instance=='P4-02':
  from research.v4.candidates.p4_02_narrow_organiser_v8_20260927.scene import Scene
  from research.v4.candidates.p4_02_straightener_heatmat_20260926.evaluator import StraightenerEvaluator as E
 elif instance=='P4-03':
  from research.v4.candidates.p4_03_render_groups_20260926.scene import VisibleGlueGunScene as Scene
  from research.v4.candidates.p4_03_gluegun_20260926.evaluator import GlueGunEvaluator as E
 else:
  from research.v4.candidates.p4_04_cold_saddle_20260926.scene import SaddleScene as Scene
  from research.v4.candidates.p4_04_soldering_iron_20260926.evaluator import SolderEvaluator as E
 if Scene.instruction!=b.instruction:raise ValueError('own Scene instruction drift')
 return b,Scene,E
