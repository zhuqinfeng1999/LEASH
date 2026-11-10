"""Actual own R5 binding-owner opt-in; only lunchbox visual XML changes."""
import importlib,json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
FOOD=ROOT/'research/v4/candidates/user_rework_r5_food_20260927/lunchbox_meal.xml'
ROWS={
 'R5-01':('r5_lighter_card_separate_v13_20260925.facade','R5CardFacade','r5_lighter_card_separate_v13_20260925.scene'),
 'R5-02':('r5_02_t1_derived_20260926.facade','R5MatchboxFacade','r5_02_t1_derived_20260926.scene'),
 'R5-03':('r5_03_t1_derived_20260926.facade','R5RazorFacade','r5_03_t1_derived_20260926.scene'),
 'R5-04':('r5_04_card_readable_v2_20260926.facade','ReadableCardFacade','r5_04_card_readable_v2_20260926.facade'),
 'R5-05':('r5_05_large_card_v2_20260926.facade','LargeCardFacade','r5_05_large_card_v2_20260926.scene'),
 'R5-06':('r5_06_vape_t1_20260926.facade','R5VapeFacade','r5_06_vape_t1_20260926.scene'),
}
def components(instance):
 if instance not in ROWS:raise ValueError('Exact six registered R5 instances only')
 clsmod,name,owner=ROWS[instance];prefix='research.v4.candidates.'
 cls=getattr(importlib.import_module(prefix+clsmod),name);module=importlib.import_module(prefix+owner)
 rows=json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances'];row=next(r for r in rows if r['id']==instance)
 if row['base_task']!='R5'or row['instructions']!=['Put the item on the left into the lunchbox.']:raise ValueError('Registered exact instruction changed')
 if sum(b.name=='lunchbox_1'for b in module.BINDINGS)!=1:raise ValueError('Own single lunchbox binding required')
 bindings=tuple(replace(b,xml=str(FOOD.relative_to(ROOT)))if b.name=='lunchbox_1'else b for b in module.BINDINGS)
 return cls,module,bindings,row
def scene_class(instance):
 original,owner,bindings,row=components(instance)
 class MealScene(original):
  def __init__(self,*args,**kwargs):
   with patch.object(owner,'BINDINGS',bindings):super().__init__(*args,**kwargs)
   if self.instance.id!=instance or self.instance.instruction!=row['instructions'][0]:
    self.close();raise RuntimeError('Actual own Scene identity mismatch')
 MealScene.__name__=instance.replace('-','')+'AuthoredMealScene'
 return MealScene
