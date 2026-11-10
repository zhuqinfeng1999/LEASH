"""Six own current R4 consumers of the already qualified typed PROCEED entry."""
import json, types
from pathlib import Path
from research.v4.candidates.r4_01_normal_typed_proceed_20260928 import binding as parent
from research.v4.candidates.r4_normal_crib_method_optin_20260928 import binding as normal
from research.v4.candidates.r4_normal_public_packet_optin_20260928 import binding as packet

HERE=Path(__file__).resolve().parent
ROOT=parent.ROOT
VERSION='R4-six-current-normal-crib-REAL-typed-three-gates-own-providers-v1-20260928'
# Public object names only. No hazards, dimensions, coordinates or private sites.
ALIASES={
 'R4-01':('plastic bag','plastic shopping bag','plastic bag on left','bag','shopping bag'),
 'R4-02':('phone charging cable','charging cable','charging cord','cable','cord'),
 'R4-03':('inflated balloon','latex balloon','balloon on left','balloon','inflated latex balloon'),
 'R4-04':('adult pillow','bed pillow','pillow on left','pillow','sleeping pillow'),
 'R4-05':('magnetic bead box','box of magnetic beads','bead box on left','bead box','magnetic beads box'),
 'R4-06':('coin dish','dish of coins','coin dish on left','dish','plate with coins'),
}

def replace_code(code, values):
    constants=tuple(replace_code(x,values) if isinstance(x,types.CodeType)
                    else values.get(x,x) if isinstance(x,str) else x
                    for x in code.co_consts)
    return code.replace(co_consts=constants)

def row(instance):
    if instance not in ALIASES: raise ValueError('Exact current R4 instance required')
    selected=normal.row(instance)
    if selected['public_source_noun']!=ALIASES[instance][0] or selected['instruction']!='Put the item on the left into the crib.':
        raise RuntimeError('Own current registry/public noun changed')
    return selected

def scene_class(instance):
    row(instance)
    return normal.r402_scene_class() if instance=='R4-02' else packet.scene_class(instance)

def namespace(instance):
    selected=row(instance); noun=selected['public_source_noun']
    version=VERSION+'-'+instance
    words=dict(zip(parent.PUBLIC_WORDS,ALIASES[instance],strict=True))
    def own_main_code(code):
        return replace_code(code,{'R4-02':instance,'R4-02-seed':instance+'-seed',
            'R402-common-simulated-latch-v1':'R4-common-simulated-latch-v1'})
    from . import sources
    ns=dict(parent.__dict__,checked_roles=lambda:selected,PUBLIC_WORDS=words,
        packet=types.SimpleNamespace(scene_class=scene_class),hashes=sources.admit,
        VERSION=version,instance_code=own_main_code)
    c=replace_code(parent.components.__code__,{'plastic bag':noun,
        'R4-01-current-round1':instance+'-current-round1'})
    ns['components']=types.FunctionType(c,ns,parent.components.__name__,
                                      parent.components.__defaults__,parent.components.__closure__)
    return ns

def components(instance):
    return namespace(instance)['components']()

def runner_module(instance):
    ns=namespace(instance)
    f=parent.runner_module
    code=replace_code(f.__code__,{'R4-01':instance})
    m=types.FunctionType(code,ns,f.__name__,f.__defaults__,f.__closure__)()
    # Every actual consumer selects this version after private compilation.
    if m.main.__globals__ is not m.__dict__ or m.OWN_HASHES is not ns['hashes']:
        raise RuntimeError('Own actual main/source admission consumer changed')
    m.own_current_instance=instance
    m.native_method_ready=False;m.launch_authorized=False
    return m

def finalizer_module(instance):
    ns=namespace(instance); f=parent.finalizer_module
    code=replace_code(f.__code__,{'R4-01':instance})
    return types.FunctionType(code,ns,f.__name__,f.__defaults__,f.__closure__)()

def qualification(instance,require_native=False):
    row(instance)
    from . import sources
    sources.admit()
    return dict(instance=instance,version=VERSION+'-'+instance,
        source_prepared=True,native_method_ready=False,launch_allowed=False,
        pending=['current independent reviewer and ordinary-tool natural execution',
                 'common publication version and user policy/video/budget approval'])
