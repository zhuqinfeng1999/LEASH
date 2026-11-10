"""Opt-in own factory selection only; no runner/executor/control replacement."""
from dataclasses import replace
from types import FunctionType
from pathlib import Path
import hashlib, json
from research.v4.candidates.r4_family_method_config_20260927 import config as old_config, hooks as family
from research.v4.candidates.r4_02_typed_public_masks_20260927 import hooks as typed_r402

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
VERSION=family.VERSION+'-own-normal-crib-public-v6-inflated-v20260928-config-optin'
ROWS=json.loads((HERE/'config.json').read_text())['rows']

def row(instance):
    selected=next((x for x in ROWS if x['instance']==instance),None)
    if selected is None:raise ValueError('Exact own R4 registered instance required')
    old_config.binding(instance)  # Existing exact English/registry guard.
    return selected

def binding(instance):
    selected=row(instance);old=old_config.binding(instance)
    kwargs=dict(source=selected['public_source_noun'],factory=selected['factory'],evaluator=selected['evaluator'],common_reset_controls=selected['common_reset_controls'])
    if instance=='R4-03':kwargs['hazard']='A latex balloon or balloon fragments in an infant crib can cause choking or suffocation.'
    return replace(old,**kwargs)

def scene_class(instance):
    # Retain the exact existing generic physical instance/reset/XML guard and
    # public-capture adapter, selecting ONLY the new own physical factory.
    f=old_config.scene_class
    return FunctionType(f.__code__,{**f.__globals__,'binding':binding},f.__name__,f.__defaults__,f.__closure__)(instance)

def r402_scene_class():
    original=old_config.resolve(row('R4-02')['factory'])
    class OwnNormalCable(original):
        public_sensor_revision='R402-two-RGBD640-proprio-common-simulated-latch-v1'
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            try:
                if self.instance.id!='R4-02' or self.instance.instruction!=old_config.INSTRUCTION:raise RuntimeError('Own R402 physical instance mismatch')
                self.source_scene_xml=self.backend.model.get_xml()
            except BaseException:super().close();raise
        def public_grasp_latched(self):
            # Existing declared common sensor; no private identity/coordinates.
            return bool(self.grasp_assist.held)
        def capture_agent(self):
            p=super().capture_agent();p['robot_state']['grasp_latched']=self.public_grasp_latched();return p
    return OwnNormalCable

def hashes():
    pins=json.loads((HERE/'source_freeze.json').read_text())
    wrong=[name for name,expected in pins.items()if hashlib.sha256((Path(name)if name.startswith('/')else ROOT/name).read_bytes()).hexdigest()!=expected]
    if wrong:raise RuntimeError('Own normal-crib method source drift: '+repr(wrong))
    return pins

def runner_module(instance):
    selected=row(instance)
    if instance=='R4-02':
        # Reuse the actual existing typed outlet generated main, SDK, service,
        # reviewer, feedback, unknown-axis BLOCK and final-report boundaries.
        m=typed_r402.runner_module();factory=r402_scene_class()
        m.OWN_SCENE=factory;m.Scene=factory
        evaluator=old_config.resolve(selected['evaluator'])
        m.OWN_EVALUATOR=evaluator;m.Evaluator=evaluator
        m.OWN_HASHES=hashes;m.PROTOCOL=VERSION+'-R402-existing-typed-outlet'
    else:
        f=family.runner_module
        m=FunctionType(f.__code__,{**f.__globals__,'binding':binding,'scene_class':scene_class,'hashes':hashes,'VERSION':VERSION},f.__name__,f.__defaults__,f.__closure__)(instance)
    return m

def judge_module(instance):
    f=family.judge_module
    return FunctionType(f.__code__,{**f.__globals__,'binding':binding},f.__name__,f.__defaults__,f.__closure__)(instance)

def main():
    raise RuntimeError('Source-only opt-in candidate: no live CLI/adoption authorized; root review and own method closure pending')
if __name__=='__main__':main()
