"""Select accepted P1 configurations with the existing task/lease Consumer."""
import ast
from types import SimpleNamespace
from unittest.mock import patch


def natural_configuration(configured, instance):
    """Keep accepted P1 physics; omit the offline script's trace/abort wrapper."""
    import os
    layout = os.environ.get('SAFEVLA_P1_LAYOUT', 'accepted')
    if layout != 'accepted':
        if layout == 'p1_single_right_arm_dev1':
            from research.v4.development.p1_single_right_arm_20261004.layout import configuration
        else:
            from research.v4.development.p1_layout_clear_endpoints_dev1.layout import configuration
        return configuration(configured, instance, layout)
    if instance != 'P1-04':
        return configured
    from research.v4.candidates.p1_04_cutter_holder_20260926.scene import P1CutterFacade
    class NaturalScene(configured.Scene):
        def step(self, action):
            return P1CutterFacade.step(self, action)
    # Do not mutate the cached configuration or its scripted acceptance Scene.
    return SimpleNamespace(**{**vars(configured), 'Scene': NaturalScene})


def runner(instance, *, runtime_overrides=None, tree_transform=None):
    from research.v4.candidates.p1_typed_initial_mask_20260927 import hooks
    from research.v4.candidates.p1_02_unknown_tilt_optin_20260927.binding import reporting_text
    from research.v4.candidates.p102_v43_actual_consumer_20260928.binding import rewrite
    from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import P102Consumer
    from research.v4.bridge.run_b0 import lite_runtime, bind_lite_main
    original_compile = hooks.compiled
    original_typed = hooks.compile_typed

    def compile_business(source, name, file, supplied):
        configured = natural_configuration(supplied['CONFIGURED'], instance)
        from research.v4.development.verified_empty_reset_20261005.initialization import wrap_configuration
        configured = wrap_configuration(configured, instance)
        supplied = {**supplied, 'CONFIGURED': configured}
        bindings = lite_runtime(instance, configured)
        bindings['V43'].Consumer = P102Consumer
        bindings['V43_VLA_CHUNK_CONTROLS'] = 5
        if runtime_overrides is not None:
            bindings.update(runtime_overrides(bindings))
        tree = ast.parse(rewrite(source))
        if instance == 'P1-06':
            # This old binding stopped both methods before any public input.
            # The current Consumer uses the shared registry: relevant rotation
            # without a reliable grasp basis stays UNRESOLVED there. It does
            # not use the old rigid E1 adapter or manufacture a vessel axis.
            main = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == 'main')
            pending_test = ast.unparse(ast.parse(
                'CONFIGURED.pending_reason and not a.preflight_only', mode='eval').body)
            main.body = [n for n in main.body if not (
                isinstance(n, ast.If) and ast.unparse(n.test) == pending_test)]
        # Only the selected own instance identifies its session. Public nouns,
        # Scene, evaluator, parser and ordinary B0 input remain configured.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 'P1-02-seed':
                node.value = instance + '-seed'
        if tree_transform is not None:
            tree = tree_transform(tree)
        return original_compile(bind_lite_main(tree, instance), name, file,
                                {**supplied, **bindings})

    def compile_typed(source, name, file, supplied):
        with patch.object(hooks, 'compiled', compile_business):
            return original_typed(reporting_text(source), name, file, supplied)

    with patch.object(hooks, 'compile_typed', compile_typed):
        module = hooks.runner_module(instance)
    module.PublicLatchToolkit = lambda base, scene: base
    module.lite_source_files = ['research/v4/bridge/p1_lite.py']
    import os
    if os.environ.get('SAFEVLA_P1_LAYOUT', 'accepted') != 'accepted':
        layout_file = ('p1_single_right_arm_20261004' if os.environ['SAFEVLA_P1_LAYOUT'] == 'p1_single_right_arm_dev1'
                       else 'p1_layout_clear_endpoints_dev1')
        module.lite_source_files.append(f'research/v4/development/{layout_file}/layout.py')
    return module
