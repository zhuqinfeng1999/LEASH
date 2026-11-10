"""Use the existing T1/P4 public reader and corner geometry in the lite entry."""
import ast
from types import FunctionType
from copy import deepcopy
from types import SimpleNamespace

from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.candidates.p401_v43_actual_consumer_20260929 import runtime as corners
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle
from research.v4.candidates.r202_v43_public_fault_types_20260929.public import (
    _capture, _fault, PublicReaderTechnicalFault,
)
from research.v4.candidates.r4_02_typed_public_masks_20260927.public import Client, DetectionMiss


def invalid_view(reader, packet, camera, name, prompt, exc):
    """Keep a failed view unknown. No trim, old box or automatic view selection."""
    import numpy as np
    span = None if exc.lower is None or exc.upper is None else exc.upper-exc.lower
    if hasattr(reader,'static'):
        reader.static.pop(name,None)
    reader.log.append(dict(frame_id=packet['frame_id'],camera=camera,name=name,prompt=prompt,
        fault=False,failure_kind='public_geometry_unresolved',error=str(exc),
        span_m=None if span is None else span.tolist(),nonfinite_values=exc.nonfinite,
        failure_axes=[] if span is None else [axis for axis,value in zip('xyz',span) if value>1.2],
        valid_pixels=getattr(exc,'valid_pixels',None),
        current_other_view_not_certified=True))
    # A single valid box must not silently replace a failed view: neither
    # same-entity identity nor relevant-region coverage has been established.
    import os
    if os.environ.get('SAFEVLA_COMPLEMENTARY_PUBLIC_VIEWS') == 'complementary-public-views-dev1-20261006':
        # The failed view is never appended. Existing union of all successful
        # same-frame boxes survives; semantic full-extent/identity and current
        # association are independent requirements, not certified here.
        reader._invalid_current_entities.discard(name)
        reader.log[-1]['current_other_view_not_certified'] = True
        reader.log[-1]['isolation_scope'] = 'failed_view_only;all_successful_views_union'
    else:
        reader._invalid_current_entities.add(name)
    # First anomaly only: keep actual failing mask and finite public metadata,
    # not a SOURCE recorder, full history, repeated inference or object graph.
    import os,json
    from pathlib import Path
    from PIL import Image
    output = os.environ.get('SAFEVLA_RUN_OUTPUT')
    if output and not getattr(reader,'_saved_geometry_anomaly',False):
        root=Path(output)/'public_geometry_anomaly'
        root.mkdir(exist_ok=False)
        mask,depth,k,t=exc.public_inputs
        rgb=np.asarray(packet['cameras'][camera]['rgb'])
        Image.fromarray(mask.astype(np.uint8)*255).save(root/'mask.png')
        overlay=rgb.copy();overlay[mask]=(rgb[mask]*.55+np.asarray([255,0,255])*.45).astype(np.uint8)
        Image.fromarray(overlay).save(root/'overlay.png')
        values=depth[mask];valid=values[np.isfinite(values)&(values>0)&(values<3)]
        ys,xs=np.nonzero(mask)
        row=dict(reader.log[-1],K=k.tolist(),T=t.tolist(),area_pixels=int(mask.sum()),
            xyxy=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None,
            valid_depth_fraction=len(valid)/max(1,len(values)),
            depth_quantiles_m=np.quantile(valid,[0,.005,.5,.995,1]).tolist() if len(valid) else None,
            component_selection='all returned mask components; none removed',
            span_limit_m=1.2,lower_world=None if exc.lower is None else exc.lower.tolist(),
            upper_world=None if exc.upper is None else exc.upper.tolist())
        (root/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
        reader._saved_geometry_anomaly=True


def declare_public_names(context, workspace, names):
    """Use the selected reader's IDs and this already acquired public frame."""
    context = dict(context)
    context.update(public_entity_names=tuple(names),
                   localized_public_entity_names=tuple(
                       box.name for box in workspace.public.visible_entities))
    context['event_description'] += (
        ' Lease entities and named no-go regions must use the exact declared '
        'public_entity_names. A declared name is not proof of localization. '
        'Keep abstract or conditional safety requirements in task_constraints '
        'or commit_conditions, not as invented named boxes. Numeric regions '
        'must be grounded in the supplied public evidence.')
    return context


def components(*, scene, evaluator, source, prompts, aliases):
    from pathlib import Path
    from research.v4.bridge.run_b0 import ROOT
    from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import compiled
    path = ROOT / 'research/v4/candidates/p4_natural_method_20260926/public_boundary.py'
    tree = ast.parse(path.read_text())

    class Names(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id in {'PROMPTS', 'ALIASES'} for t in node.targets):
                name = node.targets[0].id
                node.value = ast.Name('PUBLIC_' + name, ast.Load())
            return self.generic_visit(node)
        def visit_Constant(self, node):
            if node.value == 'iron':
                node.value = source
            return node
    tree = ast.fix_missing_locations(Names().visit(tree))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PublicWorkspaceReader')
    acquire = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
    class Acquire(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node) == 'self.scene.capture_agent()':
                return ast.copy_location(ast.Call(ast.Name('_capture', ast.Load()), [ast.Name('self', ast.Load())], []), node)
            if ast.unparse(node.func) == 'self.sam.segment':
                return ast.copy_location(ast.Call(ast.Name('_segment', ast.Load()),
                    [ast.Name(x, ast.Load()) for x in ('self', 'packet', 'camera_name', 'name', 'prompt')] + node.args, []), node)
            return node
        def visit_Return(self, node):
            return ast.parse("return _public_workspace(self, packet, tuple(float(x) for x in packet['robot_state']['eef_pos']), tuple(boxes))").body[0]
        def visit_Try(self, node):
            node = self.generic_visit(node)
            node.handlers = ast.parse("""try:
    pass
except DetectionMiss as exc:
    self.log.append({**exc.row, 'fault': False, 'failure_kind': 'structured_public_detection_miss'})
except PublicGeometryInvalid as exc:
    if visual_holding_enabled():
        invalid_view(self, packet, camera_name, name, prompt, exc)
    else:
        _fault(self, exc, stage='per_view_acquisition', frame_id=packet['frame_id'], camera=camera_name, name=name, prompt=prompt)
except Exception as exc:
    _fault(self, exc, stage='per_view_acquisition', frame_id=packet['frame_id'], camera=camera_name, name=name, prompt=prompt)
""").body[0].handlers
            return node
    cls.body[cls.body.index(acquire)] = Acquire().visit(acquire)
    from research.v4.accelerated.public_geometry import PublicGeometryInvalid
    from research.v4.leash_v43.visual_holding import enabled as visual_holding_enabled
    selected_acquire = cls.body[cls.body.index(acquire)] if acquire in cls.body else next(
        n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
    selected_acquire.body.insert(0,ast.parse('self._invalid_current_entities = set()').body[0])
    # Drop all geometry for this entity in this acquisition, even when the
    # other view returned a box. Missing remains declared and cannot be absent.
    for node in ast.walk(selected_acquire):
        if isinstance(node,ast.If) and ast.unparse(node.test)=='views':
            node.test=ast.parse('views and name not in self._invalid_current_entities',mode='eval').body

    def segment(reader, packet, camera, name, prompt, image):
        import numpy as np
        result = Client(reader.sam, dict(frame_id=packet['frame_id'], camera=camera,
                       name=name, prompt=prompt)).segment(image, text_prompt=prompt, min_score=.2)
        if (not isinstance(result.mask, np.ndarray) or result.mask.dtype != np.bool_
                or result.mask.shape != (480, 640) or result.mask_shape != (480, 640)):
            raise ValueError('Successful SAM3 mask schema malformed')
        return result

    public = compiled(ast.unparse(ast.fix_missing_locations(tree)),
        __name__ + '._public_' + source.replace(' ', '_'), path,
        dict(PUBLIC_PROMPTS=prompts, PUBLIC_ALIASES=aliases, _capture=_capture,
             _segment=segment, _fault=_fault, DetectionMiss=DetectionMiss,
             PublicGeometryInvalid=PublicGeometryInvalid,invalid_view=invalid_view,
             visual_holding_enabled=visual_holding_enabled,
             _public_workspace=common._public_workspace))
    # Reuse the T1 public alias parser for natural object descriptions.
    from research.v4.candidates.r2_02_natural_method_20260926 import public_boundary as t1
    original = t1.PublicWorkspaceReader.locate_vla_target
    public.PublicWorkspaceReader.locate_vla_target = FunctionType(
        original.__code__, {**original.__globals__, 'ALIASES': aliases},
        original.__name__, original.__defaults__, original.__closure__)
    boundary = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'P4ActionBoundary')
    execute = next(n for n in boundary.body if isinstance(n, ast.FunctionDef) and n.name == 'execute_tool')
    start = next(i for i, n in enumerate(execute.body) if ast.unparse(n) == "robot = packet['robot_state']")
    end = next(i for i, n in enumerate(execute.body) if ast.unparse(n) == 'self.geometry_workspace = transformed')
    body = deepcopy(execute.body[start:end])
    method = ast.parse('def own_configuration(self, name, arguments, public, packet): pass').body[0]
    method.body = ast.parse('lo = hi = None').body + body + ast.parse('return SimpleNamespace(public=transformed, lower=lo, upper=hi)').body
    ns = dict(vars(public), SimpleNamespace=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), ns)
    return SimpleNamespace(Scene=scene, Evaluator=evaluator, binding=SimpleNamespace(source=source),
        public_module=public, PublicWorkspaceReader=public.PublicWorkspaceReader,
        own_proposal=public.P4ActionBoundary._proposal,
        own_configuration=ns['own_configuration'], configuration_box=public.configuration_box)


def provider(components, scene, adapter, sam_client):
    return components.PublicWorkspaceReader(scene, adapter, sam_client)


class Consumer(TaskConstraints, OwnLeaseLifecycle, common.Consumer):
    resolve = corners.resolve
    _missing_payload = corners._missing_payload
    def __init__(self, *args, **kwargs):
        super().__init__(*args, provider_factory=provider, **kwargs)

    def request(self, gate, workspace, **kwargs):
        request = super().request(gate, workspace, **kwargs)
        if gate == 2:
            request = {**request, 'event_context': declare_public_names(
                request['event_context'], workspace, self.components.public_module.PROMPTS)}
        return request

    def geometry(self, resolved, workspace, lease):
        decision = corners.geometry(self, resolved, workspace, lease)
        if decision.approved or decision.system_failure:
            return decision
        if workspace.own_configuration is None and self._holding()[0] == 'UNKNOWN':
            return common.GeometryDecision(False,
                'Holding estimate is UNKNOWN from executed records and public proprioception; '
                'motion envelope is unresolved, not a missing SAM detection. A stationary '
                'OPEN may be proposed through the existing action/commit gates; only its '
                'actual stable execution can establish EMPTY. No command was dispatched.')
        for region in lease.get('no_go_regions', ()):
            if isinstance(region, str) and workspace.public.box(region) is None:
                localized = tuple(box.name for box in workspace.public.visible_entities)
                return common.GeometryDecision(False,
                    f'Reviewer-bound public region {region!r} is not localized; '
                    f'current localized exact names: {localized!r}. No region was inferred.')
        return decision
