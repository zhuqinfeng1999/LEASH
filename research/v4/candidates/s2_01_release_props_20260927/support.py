"""Pure private naming adapter to approved full physical cylinder support."""
from types import SimpleNamespace
from research.v4.candidates.s3_physical_support_predicate_20260926.predicate import basket_evidence as approved
from research.v4.accelerated.scoring import PlacementTracker

def basket_evidence(scene):
 sim=scene.env.sim;objects=scene.env.env.objects_dict
 model=sim.model
 class Model:
  def __getattr__(self,name):return getattr(model,name)
  def site_name2id(self,name):return model.site_name2id('basket_1_contain_region'if name=='s3_basket_1_contain_region'else name)
 proxy=SimpleNamespace(sim=SimpleNamespace(model=Model(),data=sim.data),backend=SimpleNamespace(objects_dict={'s3_ketchup_1':objects['ketchup_1'],'s3_basket_1':objects['basket_1']}))
 row=approved(proxy);row['revision']='S201-CC0-bottle-PolyHaven02-full-physical-interior-floor-v1';return row
class PhysicalBasketTracker(PlacementTracker):
 def __init__(self,provider):super().__init__();self.provider=provider;self.last_geometry=None
 def observe(self,**options):
  self.last_geometry=basket_evidence(self.provider());options['in_target']=self.last_geometry['in_target'];options['supported']=self.last_geometry['floor_supported'];return super().observe(**options)
