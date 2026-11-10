"""Opt-in only at original R202 real main's actual Scene/hash globals."""
from types import FunctionType
from research.v4.candidates.r2_02_natural_method_20260926 import run_method as original
from research.v4.candidates.release_r2_remaining_method_appearance_20260927 import binding as appearance
from . import freeze
VERSION='R202-real-method-optin-appearance622-v1'
def runner(*,appearance_version=None):
    if appearance_version is None:return original.main
    if appearance_version!=VERSION:raise ValueError('Unknown own real-method opt-in version')
    freeze.hashes()
    selected=appearance.own_factory('R2-02')
    ns=vars(original)|{'Scene':selected,'hashes':freeze.admit_sources,
        'PROTOCOL':original.PROTOCOL+'-appearance-'+VERSION}
    f=FunctionType(original.main.__code__,ns,__name__+'.selected_main',original.main.__defaults__,original.main.__closure__)
    f.own_appearance_version=VERSION
    return f
def describe(*,appearance_version=None):
    f=runner(appearance_version=appearance_version)
    return {'own_instance':'R2-02','opt_in':appearance_version is not None,
        'same_original_main_code':f.__code__ is original.main.__code__,
        'selected_Scene_module':f.__globals__['Scene'].__module__,
        'selected_hash_function':f.__globals__['hashes'].__module__+'.'+f.__globals__['hashes'].__name__,
        'protocol':f.__globals__['PROTOCOL'],'instruction':f.__globals__['INSTRUCTION'],
        'current_main_invoked':False,'models_services_or_Scene_started':False,
        'live_launch_or_common_adoption_authorized':False}
