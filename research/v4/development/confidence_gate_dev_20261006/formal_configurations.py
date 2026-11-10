"""Formal selection of existing accepted physical factories; no old gates.

Missing qualification remains missing. Selection itself performs no reset,
model request, or inference of EMPTY from private state.
"""
from importlib import import_module
from unittest.mock import patch
from .configurations import _configured, configuration, registry_row


def select(instance):
    if instance in ('P1-01', 'P1-02'):
        from .runner import p1_configuration
        return p1_configuration(instance)
    if instance == 'P1-03':
        import hashlib,json
        from pathlib import Path
        from .p103_bridge_grasp_annotation_20261014 import configure, VERSION
        from .runner import p1_configuration
        root=Path(__file__).resolve().parents[4]
        receipt=root/'research/v4/results/formal_confidence_20261008/user_seed0_review_20261014/root/p103_physical_qualification.json'
        proof=json.loads(receipt.read_text()) if receipt.exists() else {}
        annotation=root/'research/v4/development/confidence_gate_dev_20261006/p103_bridge_grasp_annotation_20261014.py'
        if (proof.get('qualified') is not True or proof.get('qualified_formal_seeds')!=[0,1,2]
                or proof.get('physical_annotation_version')!=VERSION
                or proof.get('physical_annotation_sha256')!=hashlib.sha256(annotation.read_bytes()).hexdigest()
                or proof.get('online_dynamic_truth') is not False):
            raise ValueError('P1-03 physical grasp-chain acceptance unavailable for current recipe')
        for relative,digest in proof['source_recipe_sha256'].items():
            if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:
                raise ValueError('P1-03 qualified physical recipe has changed')
        return configure(p1_configuration(instance))
    if not instance.startswith('R') or instance.endswith('-01'):
        return configuration(instance)
    if instance.startswith('R1-'):
        from research.v4.bridge.r1_family_lite import SCENES
        package, name, source, target = SCENES[instance]
        parent = getattr(import_module('research.v4.candidates.'+package+'.scene'), name)
        from research.v4.candidates.r1_02_t1_derived_20260926.offline_evaluator import Evaluator
    elif instance.startswith('R2-'):
        if instance == 'R2-02':
            from research.v4.candidates.r2_02_natural_method_20260926 import run_method as m
            original, Evaluator = m.Scene, m.Evaluator
        elif instance == 'R2-03':
            from research.v4.candidates.r2_03_natural_method_20260926 import run_method as m
            original, Evaluator = m.T1R2TealightScene, m.Evaluator
        else:
            from research.v4.candidates.r2_remaining_method_20260926 import run_method as m
            original, Evaluator = m.CONFIG[instance][:2]
        class parent(original):
            def __init__(self, *args, **kwargs):
                from robosuite.utils.binding_utils import MjSim
                from research.v4.candidates.panda_release_factory_optin_20260927.helper import source_identity
                compile_model = MjSim.from_xml_string
                with patch.object(MjSim, 'from_xml_string', classmethod(
                        lambda cls, xml: compile_model(source_identity(xml)))):
                    super().__init__(*args, **kwargs)
            @property
            def backend(self):
                return self._env if hasattr(self, '_env') else self.env.env
            @backend.setter
            def backend(self, value): self._env = value
        source, target = None, None
    elif instance.startswith('R3-'):
        from research.v4.bridge.r3_family_lite import scene_type
        from research.v4.tasks.r3.evaluator import Evaluator as OriginalEvaluator
        parent = scene_type(instance)
        class Evaluator(OriginalEvaluator):
            def __init__(self):
                # The legacy template loader only declares R3-01. Derived
                # factories have their own accepted actual scene bindings;
                # adopt that binding at first observe, not a base-task proxy.
                self.instance=None
                self.first={'HARM':None,'UNSAFE_COMMIT':None}
                self.samples=0; self.last=None
                self._ever=False; self._witness_count=0; self._witness_complete=True
            def observe(self,scene):
                if scene.instance.id!=instance:
                    raise ValueError('R3 formal evaluator/actual scene mismatch')
                if self.instance is None:self.instance=scene.instance
                row=super().observe(scene)
                from .r3_attachment_observation import _current_dangerous_assist
                held,reason=_current_dangerous_assist(scene)
                if type(held) is not bool:self._witness_complete=False
                else:self._ever |= held; self._witness_count+=1
                return row
            def finish(self,*args,**kwargs):
                raw=super().finish(*args,**kwargs)
                return dict(raw,ever_attached=self._ever if self._witness_complete else None,
                    dangerous_item_assist_observation='complete' if self._witness_complete else 'unavailable',
                    dangerous_item_assist_samples=self._witness_count)
        source, target = None, None
    elif instance.startswith('R4-'):
        from research.v4.candidates.r4_normal_crib_method_optin_20260928 import binding
        row = binding.row(instance)
        parent = binding.old_config.resolve(row['factory'])
        Evaluator = binding.old_config.resolve(row['evaluator'])
        source, target = row['public_source_noun'], row['public_target_noun']
    elif instance.startswith('R5-'):
        from research.v4.candidates.user_rework_r5_food_bindings_20260927.binding import scene_class
        from research.v4.candidates.r5_derived_natural_method_20260926.offline_evaluator import Evaluator
        parent = scene_class(instance)
        source, target = None, 'lunchbox'
    else:
        raise ValueError('No current physical factory for '+instance)
    return _configured(instance, parent, Evaluator, source=source, target=target,
                       instructions=registry_row(instance)['instructions'])
