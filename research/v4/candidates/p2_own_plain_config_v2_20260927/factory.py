"""One causal hook correction: actual P201 proposal local is public."""
import ast
from pathlib import Path
from research.v4.candidates.p2_own_plain_config_20260927 import factory as old
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import hooks as guard
text=Path(old.__file__).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='components');fn=ast.get_source_segment(text,node)
fn=guard.replace_once(fn,'proposal = CONFIGURED_PUBLIC.proposal(name, input_dict, workspace, self.locate_vla_target)','proposal = CONFIGURED_PUBLIC.proposal(name, input_dict, public, self.locate_vla_target)')
ns={**vars(old),'PACKAGE':__package__,'HERE':Path(__file__).resolve().parent}
exec(compile(fn,__file__,'exec'),ns);components=ns['components']
