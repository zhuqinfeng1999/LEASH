"""E305 current accepted own scene; preparation only, no live authorization."""
from dataclasses import replace
from research.v4.candidates.e3_family_current_method_config_20260927.config import get

INSTANCE = 'E3-05'
CONFIG = replace(get(INSTANCE),
    factory='research.v4.candidates.user_rework_e305_seated_reach_v5_20260927.scene:Scene',
    acceptance='research/v4/results/e3_05_user_seated_palm_v42_instance_acceptance_20260927.json',
    qualification='research/v4/candidates/user_rework_e305_seated_reach_v5_20260927/qualification_seed0to9.json')
NATIVE_READY = False
LAUNCH_AUTHORIZED = False
