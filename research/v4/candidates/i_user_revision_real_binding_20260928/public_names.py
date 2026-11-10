"""Original public-name code with the registered knife branch made explicit."""
import ast
from pathlib import Path
from research.v4.candidates.i_own_plain_config_20260927 import public_names as parent
from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import compiled
from research.v4.candidates.e1_02_plain_rpent_entry_20260927.hooks import replace_once
from . import config

path = Path(parent.__file__)
text = replace_once(path.read_text(), 'from .config import binding', 'binding = CURRENT_BINDING')
text = replace_once(text,
    "else:aliases.update({'can':'metal can','soup can':'metal can','steel can':'metal can'})",
    "elif b.other=='metal can':aliases.update({'can':'metal can','soup can':'metal can','steel can':'metal can'})\n elif b.other=='knife':aliases.update({'kitchen knife':'knife'})\n else:raise ValueError('Unbound own protected object')")
selected = compiled(text, __package__ + '._public_names', path, {'CURRENT_BINDING': config.binding})
for name in ('PARTS', 'canonical', 'expanded', 'bind_task', 'perception_queries', 'public_box', 'locate_goal'):
    globals()[name] = getattr(selected, name)
