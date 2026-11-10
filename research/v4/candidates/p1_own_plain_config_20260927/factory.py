"""Scoped production bindings; original controller/core remain unchanged."""
import ast
from functools import lru_cache
from types import SimpleNamespace
from dataclasses import replace
from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import compiled
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import hooks as parent
from research.v4.candidates.e1_02_plain_rpent_entry_20260927.factory import PlainRPentToolkit
from research.v4.accelerated.e1_gated_adapter import E1GatedAdapter
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import _segment_hits_box
from . import config
PACKAGE='research.v4.candidates.p1_own_plain_config_20260927'

@lru_cache(maxsize=6)
def components(instance):
 b,Scene,Evaluator=config.components(instance)
 path=parent.HERE/'bindings.py';text=path.read_text();parent.source_hashes()
 text=parent.replace_once(text,'from research.v4.candidates.e1_future_public_binding_20260927 import public_names as names',f'from {PACKAGE} import public_names as names')
 text=parent.replace_once(text,"INSTANCE = 'E1-02'\nSOURCE = 'cleaver'\nTARGET = 'cutting board'",'INSTANCE = CONFIG.instance\nSOURCE = CONFIG.source\nTARGET = CONFIG.target')
 text=parent.replace_once(text,"raise ValueError('held cleaver full public box unavailable')","raise ValueError('held requested payload public box unavailable')")
 bound=compiled(text,PACKAGE+'._binding_'+instance.replace('-','_'),path,{'CONFIG':b})
 tp='research/v4/accelerated/e1_gated_toolkit.py';text=parent.parent(tp)
 text=parent.replace_once(text,'from research.v4.accelerated.e1_name_binding import bind_motion_names','bind_motion_names = CONFIGURED_BIND_MOTION')
 text=parent.replace_once(text,'proposal = proposal_for_e1(name, input_dict, workspace,\n                                       self.locate_vla_target)','proposal = self._public_proposal(name, input_dict, workspace,\n                                       self.locate_vla_target)')
 text=parent.replace_once(text,'        self._pick_operation: tuple[str, str] | None = None\n','')
 text=parent.replace_span(text,'            if (self._pick_operation is not None and\n','            decision = self.gate.check(proposal)\n','')
 tree=ast.parse(text);cuts=[]
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in ('E1HeldSweep','proposal_for_e1'):cuts.append(n)
  if isinstance(n,ast.ClassDef)and n.name=='E1GatedToolkit':
   cuts.extend(x for x in n.body if isinstance(x,ast.FunctionDef)and x.name=='operation_scope')
 if len(cuts)!=3:raise RuntimeError('fixed envelope/macro AST mismatch')
 lines=text.splitlines(keepends=True)
 for n in sorted(cuts,key=lambda n:n.lineno,reverse=True):
  start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1;del lines[start:n.end_lineno]
 text=''.join(lines)
 text=parent.replace_once(text,'held_object_descent_before_E1_5cm_critical','held_descent_before_P1_landing')
 module=compiled(text,PACKAGE+'._toolkit_'+instance.replace('-','_'),parent.ROOT/tp,{'CONFIGURED_BIND_MOTION':bound.motion_names})
 class Toolkit(module.E1GatedToolkit):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self._public_proposal=bound.proposal;generic=self.gate.check
   def p1_check(proposal):
    ws=self._pending_workspace
    if ws.evidence.robot.hold_estimate=='held':
     if proposal.name in ('rotate_wrist','rotate_pitch','move_pose'):
      return self.gate._deny('Held rotation sweep not qualified in this public-box protocol','Preserve current orientation or ask; no rigid-shape or container-axis assumption.',reviewer_called=False)
     path=proposal.motion_waypoints
     if proposal.name=='vla_act':path=(ws.eef_xyz,proposal.vla_target_xyz)if proposal.vla_target_xyz else ()
     if path:
      radius=proposal.gripper_radius_m+proposal.held_object_radius_m
      from .public_names import PARTS
      for part in sorted(PARTS):
       box=ws.box(part)
       if box is None:return self.gate._deny('Human part public mask unavailable','Refresh the two public views; do not execute.',reviewer_called=False)
       zone=box.expanded(radius);z=(zone.lower[2]+zone.upper[2])/2
       if any(_segment_hits_box((a[0],a[1],z),(e[0],e[1],z),zone)for a,e in zip(path,path[1:])):
        return self.gate._deny('Held payload public sweep crosses '+part+' at any height','Keep all path segments outside the public human XY footprint plus observed payload/EEF envelope; re-propose using ordinary primitives.',reviewer_called=False)
    return generic(proposal)
   self.gate.check=p1_check
  def operation_scope(self,*args,**kwargs):raise ValueError('No pick macro in this common ordinary-RPent protocol')
 return SimpleNamespace(binding=b,Scene=Scene,Evaluator=Evaluator,bindings=bound,
  PublicWorkspaceFactory=bound.PublicWorkspaceFactory,PlainRPentToolkit=PlainRPentToolkit,
  P1GatedToolkit=Toolkit,P1MotionReviewer=module.E1MotionReviewer,
  E1GatedAdapter=E1GatedAdapter,cpu_call_path_ready=b.pending is None,
  pending_reason=b.pending,native_mask_review_ready=False,method_launch_authorized=False,
  injected_bindings_sha256=bound.injected_source_sha256,injected_toolkit_sha256=module.injected_source_sha256)
