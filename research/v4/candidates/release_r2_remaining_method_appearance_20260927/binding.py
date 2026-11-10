"""Exact existing own R2 methods, separate singlecompile opt-in, not R201547."""
from types import SimpleNamespace
from research.v4.candidates.r2_02_natural_method_20260926 import run_method as r202
from research.v4.candidates.r2_03_natural_method_20260926 import run_method as r203
from research.v4.candidates.r2_remaining_method_20260926 import run_method as remaining
from research.v4.candidates.release_appearance_composed_factory_20260927 import helper as appearance
VERSION='R2-own02to05-singlecompile-release-Panda142-no-person-v1'
INSTANCES=('R2-02','R2-03','R2-04','R2-05')
def own(instance):
    if instance=='R2-02':S,E,entry=r202.Scene,r202.Evaluator,r202
    elif instance=='R2-03':S,E,entry=r203.T1R2TealightScene,r203.Evaluator,r203
    elif instance in ('R2-04','R2-05'):S,E,*_=remaining.CONFIG[instance];entry=remaining
    else:raise ValueError('Only own R202–205 allowed')
    from research.v4.candidates.r2_02_public_reset_20260926.scene import R2IncensePublicResetScene
    from research.v4.candidates.r2_03_t1_derived_20260926.scene import T1R2TealightScene
    from research.v4.candidates.r2_04_t1_derived_20260926.scene import T1R2CharcoalScene
    from research.v4.candidates.r2_05_t1_derived_20260926.scene import T1R2DrawerScene
    expected=dict(zip(INSTANCES,(R2IncensePublicResetScene,T1R2TealightScene,T1R2CharcoalScene,T1R2DrawerScene)))
    if S is not expected[instance]or S.task_id!=instance:raise RuntimeError('Own factory identity drift')
    return SimpleNamespace(instance=instance,Scene=S,Evaluator=E,instruction=S.instruction,entry=entry,
        Adapter=r202.R2V42Adapter,OursAdapter=r202.E1GatedAdapter,
        PublicLatchToolkit=r202.PublicLatchToolkit,method_native_ready_claim=False)
def construct(instance,seed=0,*,camera_observations=False,render_gpu_device_id=-1,appearance_version=None):
    c=own(instance);kw=dict(camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
    if appearance_version is None:return c.Scene(seed,**kw)
    if appearance_version!=VERSION:raise ValueError('Unknown own singlecompile version')
    s=appearance.construct(c.Scene,seed,appearance_version=appearance.APPEARANCE_VERSION,**kw)
    s.own_method_appearance_version=VERSION;return s
def own_factory(instance):
    own(instance)
    def factory(seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        return construct(instance,seed,appearance_version=VERSION,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
    return factory
