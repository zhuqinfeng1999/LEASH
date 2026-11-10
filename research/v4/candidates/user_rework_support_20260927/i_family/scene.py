"""Own I text/label factories with one common real prop layout/support version."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.i_base_floor_scripts_20260926 import scene as bases
from research.v4.candidates.i_t7_candle_can_derived_20260926 import scene as derived
from research.v4.candidates.i4_named_label_pose_v2_20260926 import scene as candle_label
from research.v4.candidates.i4_can_label_pose_v3_20260926 import scene as can_label
from research.v4.candidates.user_rework_support_20260927.i1_layout_v3 import scene as qualified
from research.v4.candidates.i_common_release_ketchup_20260926 import scene as common
H=Path(__file__).resolve().parent;ROOT=H.parents[4]
IDS=tuple(f'I{family}-{variant:02d}'for family in range(1,5)for variant in range(1,4))
BASE_COMMON_CREATE=common.create
TAGGED_CANDLE=ROOT/'research/v4/candidates/i4_named_label_pose_v2_20260926/generated/I4-02/tagged_object.xml'

def layout(bindings):
    result=[]
    for b in bindings:
        if b.name=='paper_bin_1':b=replace(b,slot='B',xy_offset=(0.,-.26))
        elif b.name=='blunt_tool_1':b=replace(b,slot='C',xy_offset=(.15,-.08))
        elif b.name=='soup_can_1':b=replace(b,slot='C',xy_offset=(.28,-.18))
        elif b.name=='lit_candle_1':
            path=Path(b.xml);path=path if path.is_absolute()else ROOT/path
            if path.resolve()==TAGGED_CANDLE.resolve():b=replace(b,xml=str(H/'tagged_candle_planar.xml'))
            elif 'tagged_object'in path.name:raise ValueError('Unknown own candle tag source; fail closed')
            else:b=replace(b,xml=str(H.parent/'lit_candle_planar.xml'))
        result.append(b)
    return tuple(result)

def env_layout(original):
    class LayoutEnv(original):
        @staticmethod
        def person_builder(arena):
            original.person_builder(arena)
            fixture=arena.root.find(".//body[@name='i1_microwave_fixture']")
            if fixture is None:raise RuntimeError('Exact own microwave missing')
            fixture.set('pos','-.22 .29 .9')
    return LayoutEnv

def create(instance,seed,**options):
    if instance not in IDS:raise ValueError(instance)
    if instance=='I1-01':return qualified.create(seed,**options)
    if instance.endswith('-01'):
        module=bases.other.BASES[instance]
        attr={'I2-01':'I2StraightRightEnv','I3-01':'_I3Env','I4-01':'I4HandleLabelEnv'}[instance]
        with patch.object(module,'BINDINGS',layout(module.BINDINGS)),patch.object(module,attr,env_layout(getattr(module,attr))):
            return bases.create(instance,seed,**options)
    # Let the exact own text/tag/park factory build its own selected environment,
    # then change only the declared common bindings and fixed-device placement.
    def configured(seed,**kw):
        with patch.object(common,'I1StraightRightEnv',env_layout(common.I1StraightRightEnv)),patch.object(common,'BINDINGS',layout(common.BINDINGS)):
            return BASE_COMMON_CREATE(seed,**kw)
    own=candle_label.create if instance=='I4-02'else can_label.create if instance=='I4-03'else derived.create
    with patch.object(common,'create',configured):return own(instance,seed,**options)
