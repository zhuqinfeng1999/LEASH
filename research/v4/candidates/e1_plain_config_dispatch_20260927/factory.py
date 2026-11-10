"""One configuration binder over frozen E1 factories; no simulator construction."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
from functools import lru_cache
from types import SimpleNamespace
import hashlib
import ast
from pathlib import Path
import sys
from types import ModuleType

from research.v4.candidates.e1_future_public_binding_20260927 import config, public_names
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import hooks as parent
from research.v4.candidates.e1_02_plain_rpent_entry_20260927.factory import PlainRPentToolkit
from research.v4.candidates.e1_02_plain_rpent_v2_20260927.signals import (
    MissingPublicEvidence as ParentMissing, initial_missing_has_other_fault)
from research.v4.accelerated.e1_gated_adapter import E1GatedAdapter

HERE = Path(__file__).resolve().parent
PACKAGE = 'research.v4.candidates.e1_plain_config_dispatch_20260927'


def compiled(source, name, file, supplied=None):
    module = ModuleType(name)
    module.__file__ = str(file); module.__package__ = PACKAGE
    module.__dict__.update(supplied or {})
    module.injected_source_sha256 = hashlib.sha256(source.encode()).hexdigest() if source_custody_enabled() else None
    sys.modules[name] = module
    if not source_custody_enabled() and '_runner_' in name:
        import os
        if os.environ.get('SAFEVLA_LEASH_INSTANCE') == 'E1-01':
            from research.v4.bridge.run_b0 import bind_lite_main, lite_runtime
            from research.v4.candidates.e101_v43_actual_main_binding_20260929.binding import transformed_tree
            module.__dict__.update(lite_runtime('E1-01', components('E1-01')))
            source = bind_lite_main(transformed_tree(source), 'E1-01')
    try:exec(compile(source,str(file),'exec'),module.__dict__)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module


class PendingContainerAdapter:
    def __init__(self,*args,**kwargs):
        raise RuntimeError('Pending public container tilt binding; no rigid-tool bypass')


@lru_cache(maxsize=6)
def components(instance):
    b, Scene, Evaluator = config.resolve_components(instance)
    suffix=instance.replace('-','_')
    source_path=Path(parent.HERE)/'bindings.py'
    # Parent bindings are already source-pinned by the immutable v1 union.
    if source_custody_enabled():
        parent.source_hashes()
    source=source_path.read_text()
    source=parent.replace_once(source,"INSTANCE = 'E1-02'\nSOURCE = 'cleaver'\nTARGET = 'cutting board'",
        'INSTANCE = CONFIG.instance\nSOURCE = CONFIG.source\nTARGET = CONFIG.target')
    source=parent.replace_once(source,"raise ValueError('held cleaver full public box unavailable')",
        "raise ValueError('held public payload box unavailable')")
    bound=compiled(source,PACKAGE+'._bindings_'+suffix,source_path,{'CONFIG':b})
    toolkit_path='research/v4/accelerated/e1_gated_toolkit.py'
    text=parent.parent(toolkit_path)
    text=parent.replace_once(text,
        'from research.v4.accelerated.e1_name_binding import bind_motion_names',
        'bind_motion_names = CONFIGURED_BIND_MOTION')
    text=parent.replace_once(text,
        'proposal = proposal_for_e1(name, input_dict, workspace,\n                                       self.locate_vla_target)',
        'proposal = self._public_proposal(name, input_dict, workspace,\n                                       self.locate_vla_target)')
    # Remove obsolete, unreachable fixed-knife envelope/proposal definitions.
    tree=ast.parse(text)
    obsolete=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and
              n.name in {'E1HeldSweep','proposal_for_e1'}]
    if len(obsolete)!=2:raise RuntimeError('legacy fixed-envelope occurrence mismatch')
    lines=text.splitlines(keepends=True)
    for node in reversed(obsolete):
        start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
        del lines[start:node.end_lineno]
    text=''.join(lines)
    toolmod=compiled(text,PACKAGE+'._toolkit_'+suffix,parent.ROOT/toolkit_path,
                     {'CONFIGURED_BIND_MOTION':bound.motion_names})
    class ConfiguredToolkit(toolmod.E1GatedToolkit):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._public_proposal=bound.proposal
        def operation_scope(self,*args,**kwargs):
            raise ValueError('No pick macro is registered in this common protocol')
    class MissingPublicEvidence(ParentMissing):
        def __init__(self,missing,perception):
            if not missing or any(x not in bound.NAMES for x in missing):
                raise ValueError('invalid configured initial missing-entity cause')
            self.missing=tuple(sorted(set(missing)));self.perception=tuple(perception)
            Exception.__init__(self,'Valid initial public capture had no mask for: '+', '.join(self.missing))
    return SimpleNamespace(binding=b, Scene=Scene, Evaluator=Evaluator,
        PublicWorkspaceFactory=bound.PublicWorkspaceFactory,
        E1GatedToolkit=ConfiguredToolkit,E1MotionReviewer=toolmod.E1MotionReviewer,
        PlainRPentToolkit=PlainRPentToolkit,
        E1GatedAdapter=E1GatedAdapter if b.category=='rigid_tool' else PendingContainerAdapter,
        bindings=bound,MissingPublicEvidence=MissingPublicEvidence,
        initial_missing_has_other_fault=initial_missing_has_other_fault,
        injected_bindings_sha256=bound.injected_source_sha256,
        injected_toolkit_sha256=toolmod.injected_source_sha256,
        cpu_call_path_ready=b.category=='rigid_tool',
        native_mask_review_ready=instance=='E1-02',
        method_launch_authorized=False,
        pending_reason=(None if b.category=='rigid_tool' else
            'Public container tilt state/transition must reach generic commit review; current rigid adapter would incorrectly bypass it.'))
