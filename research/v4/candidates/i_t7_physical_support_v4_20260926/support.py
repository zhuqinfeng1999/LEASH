"""Reuse approved S3 authored floor/full-footprint predicate, private names only."""
from types import SimpleNamespace
from research.v4.accelerated.scoring import PlacementTracker
from research.v4.candidates.s3_physical_support_predicate_20260926.predicate import basket_evidence as original_evidence
class ModelNames:
    def __init__(self,model):self.original=model
    def __getattr__(self,name):return getattr(self.original,name)
    def site_name2id(self,name):
        return self.original.site_name2id('basket_1_contain_region'if name=='s3_basket_1_contain_region'else name)
def basket_evidence(scene):
    backend=scene.backend;sim=backend.sim
    objects=dict(backend.objects_dict)
    objects['s3_ketchup_1']=objects['ketchup_1'];objects['s3_basket_1']=objects['basket_1']
    proxy=SimpleNamespace(sim=SimpleNamespace(model=ModelNames(sim.model),data=sim.data),backend=SimpleNamespace(objects_dict=objects))
    result=original_evidence(proxy);result['revision']='i-shared-full-bottle-authored-interior-floor-support-v1'
    return result
class PhysicalBasketTracker(PlacementTracker):
    def __init__(self,provider):super().__init__();self.provider=provider;self.last_geometry=None
    def observe(self,**kwargs):
        self.last_geometry=basket_evidence(self.provider())
        kwargs['in_target']=bool(self.last_geometry['in_target']);kwargs['supported']=bool(self.last_geometry['floor_supported'])
        return super().observe(**kwargs)
