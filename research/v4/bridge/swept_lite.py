"""Existing public-box acquisition and task-specific swept geometry in lite."""
import ast
import inspect
import textwrap
from types import FunctionType, SimpleNamespace

from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import bindings as acquisition
from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.candidates.p302_v43_actual_consumer_20260929.runtime import P302PublicConsumer
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle
from research.v4.candidates.r202_v43_public_fault_types_20260929.public import _capture, _fault
from research.v4.candidates.r4_02_typed_public_masks_20260927.public import Client, DetectionMiss


def public_factory(prompts, locate):
    """Keep original two-view union and 3 cm bounds, with typed actual SAM."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(acquisition.PublicWorkspaceFactory.__call__)))
    class Input(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node) == 'self.scene.capture_agent()':
                return ast.copy_location(ast.parse('_capture(self)', mode='eval').body, node)
            if ast.unparse(node.func) == 'names.public_box':
                node.func = ast.Name('_box', ast.Load())
                node.args = [ast.Name('self', ast.Load()), ast.Name('packet', ast.Load()), ast.Name('noun', ast.Load())]
            return node
        def visit_Return(self, node):
            return ast.parse('return _public_workspace(self, packet, eef, tuple(boxes))').body[0]
        def visit_Try(self, node):
            node = self.generic_visit(node)
            node.handlers = ast.parse("""try:
    pass
except DetectionMiss as exc:
    self.log.append({**exc.row, 'fault': False, 'failure_kind': 'structured_public_detection_miss'})
except Exception as exc:
    _fault(self, exc, stage='per_view_acquisition', frame_id=frame, camera=view, name=noun, prompt=PROMPTS[noun])
""").body[0].handlers
            return node
    def box(reader, packet, noun, *, camera, sam3_client):
        from research.v4.accelerated.public_geometry import sam_public_box
        camera_name = next(name for name, value in packet['cameras'].items() if value is camera)
        client = Client(sam3_client, dict(frame_id=packet['frame_id'], camera=camera_name,
                                        name=noun, prompt=prompts[noun]))
        return sam_public_box(name=noun, prompt=prompts[noun], camera=camera, sam3_client=client)
    ns = {**acquisition.__dict__, 'PROMPTS':prompts, '_capture':_capture,
          '_box':box, '_fault':_fault, 'DetectionMiss':DetectionMiss,
          '_public_workspace':common._public_workspace}
    exec(compile(ast.fix_missing_locations(Input().visit(tree)), __file__, 'exec'), ns)
    return type('TypedPublicWorkspaceFactory', (acquisition.PublicWorkspaceFactory,),
                {'__call__':ns['__call__'], 'locate_vla_target':staticmethod(locate),
                 'public_entity_names': tuple(prompts)})


def provider(components, scene, adapter, sam_client):
    return components.PublicWorkspaceFactory(scene, adapter, sam_client)


class Consumer(TaskConstraints, OwnLeaseLifecycle, common.Consumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, provider_factory=provider, **kwargs)


class P3Consumer(TaskConstraints, OwnLeaseLifecycle, P302PublicConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, provider_factory=provider, **kwargs)


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    from research.v4.bridge.t1_lite import runner as main_runner, public_capture
    if instance.startswith('P2-'):
        if instance == 'P2-01':
            from research.v4.candidates.p2_01_revised_binding_20260928 import factory as base
        else:
            from research.v4.candidates.p2_own_plain_config_v2_20260927 import factory as base
        from research.v4.candidates.p2_own_plain_config_20260927 import public_names
        original = base.components(instance)
        prompts = public_names.perception_queries(instance)
        locate = original.PublicWorkspaceFactory.locate_vla_target
        selected_consumer = Consumer
    elif instance.startswith('P3-'):
        from research.v4.candidates.p3_own_method_dispatch_20260927 import factory as base
        original = base.components(instance)
        b = original.binding
        prompts = {b.source:b.source_prompt, b.protected:b.protected_prompt, b.target:b.target}
        locate = original.locate
        selected_consumer = P3Consumer
    else:
        raise ValueError('unregistered swept task')
    b = original.binding
    parent = original.Scene
    class Scene(parent):
        capture_agent = public_capture
    Scene.instruction = original.INSTRUCTION if hasattr(original, 'INSTRUCTION') else b.instruction
    sweep = type('OwnObservedPayloadSweep', (acquisition.ObservedPayloadSweep,), {
        '__init__':FunctionType(acquisition.ObservedPayloadSweep.__init__.__code__,
            {**acquisition.__dict__, 'SOURCE':b.source}, '__init__')})
    configured = SimpleNamespace(**{**vars(original), 'Scene':Scene,
        'PublicWorkspaceFactory':public_factory(prompts, locate),
        'bindings':SimpleNamespace(ObservedPayloadSweep=sweep)})
    module = main_runner(instance, scene=Scene, evaluator=original.Evaluator,
        source=b.source, prompts=prompts, aliases={}, configured=configured,
        consumer_class=selected_consumer,
        runtime_overrides=runtime_overrides, tree_transform=tree_transform)
    module.lite_source_files = ['research/v4/bridge/t1_lite.py', 'research/v4/bridge/swept_lite.py']
    if instance.startswith('P3-'):
        module.lite_source_files.append('research/v4/candidates/p302_v43_actual_consumer_20260929/runtime.py')
    return module
