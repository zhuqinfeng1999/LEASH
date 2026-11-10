"""Thin original P2 toolkit injection; translated public bounds, no tip oracle."""
import ast,hashlib,json
from functools import lru_cache
from types import SimpleNamespace
from pathlib import Path
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import hooks as h
from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import compiled
from research.v4.candidates.e1_02_plain_rpent_entry_20260927.factory import PlainRPentToolkit
from . import config
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PACKAGE='research.v4.candidates.p2_own_plain_config_20260927'
@lru_cache(maxsize=6)
def components(instance):
 b,S,E=config.components(instance)
 path=h.HERE/'bindings.py';text=path.read_text()
 text=h.replace_once(text,'from research.v4.candidates.e1_future_public_binding_20260927 import public_names as names',f'from {PACKAGE} import public_names as names')
 text=h.replace_once(text,"INSTANCE = 'E1-02'\nSOURCE = 'cleaver'\nTARGET = 'cutting board'",'INSTANCE = CONFIG.instance\nSOURCE = CONFIG.source\nTARGET = CONFIG.target')
 text=h.replace_once(text,"raise ValueError('held cleaver full public box unavailable')","raise ValueError('held requested payload public box unavailable')")
 bound=compiled(text,PACKAGE+'._public_'+instance.replace('-','_'),path,{'CONFIG':b})
 path=ROOT/'research/v4/candidates/p2_v42_t3_20260925/gate_method_v1.py';text=path.read_text();tree=ast.parse(text)
 # Replace ONLY original noun configuration functions. The actual P2 gate
 # lifecycle, whole visible-box displacement check and hold/cancel stay intact.
 start=text.index('HUMAN_PARTS = ');end=text.index('class P2MotionReviewer:')
 text=text[:start]+f'HUMAN_PARTS = CONFIGURED_PARTS\nPROMPTS = CONFIGURED_PUBLIC.PROMPTS\nbind_task_names = CONFIGURED_PUBLIC.task_names\nbind_motion_names = CONFIGURED_PUBLIC.motion_names\nforbidden_parts = lambda value: CONFIGURED_EXPANDED(value)\n\n'+text[end:]
 # Original literals referred to the requested payload role; neutral noun now.
 text=text.replace('"knife"','CONFIGURED_SOURCE').replace("'knife'",'CONFIGURED_SOURCE')
 old='proposal = proposal_for_p3(name, input_dict, workspace,\n                                       self.locate_vla_target,\n                                       held_sweep=PublicKnifeSweptRadius())'
 if old not in text:
  # Match actual original source exactly; no permissive broad fallthrough.
  node=next(n for n in ast.walk(ast.parse(text))if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='proposal'for x in n.targets)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='proposal_for_p3')
  old=ast.get_source_segment(text,node)
 text=h.replace_once(text,old,'proposal = CONFIGURED_PUBLIC.proposal(name, input_dict, workspace, self.locate_vla_target)')
 # VLA operated role is returned by the common public intent helper, not any
 # cross-instance "blade/handle" keyword. Keep the existing source box test.
 text=h.replace_once(text,'("knife", "blade", "handle")','(CONFIGURED_SOURCE,)')if '("knife", "blade", "handle")'in text else text
 text=text.replace('(CONFIGURED_SOURCE, "blade", "handle")','(CONFIGURED_SOURCE,)')
 module=compiled(text,PACKAGE+'._gate_'+instance.replace('-','_'),path,{'CONFIGURED_SOURCE':b.source,'CONFIGURED_PARTS':__import__(PACKAGE+'.public_names',fromlist=['PARTS']).PARTS,'CONFIGURED_PUBLIC':bound,'CONFIGURED_EXPANDED':lambda value:__import__(PACKAGE+'.public_names',fromlist=['expanded']).expanded(instance,value)})
 return SimpleNamespace(binding=b,Scene=S,Evaluator=E,bindings=bound,PublicWorkspaceFactory=bound.PublicWorkspaceFactory,PlainRPentToolkit=PlainRPentToolkit,P2GatedToolkit=module.P2GatedToolkit,P2MotionReviewer=module.P2MotionReviewer,pending_reason=b.pending,native_ready=False,launch_authorized=False,injected_public_sha256=bound.injected_source_sha256,injected_gate_sha256=module.injected_source_sha256)
