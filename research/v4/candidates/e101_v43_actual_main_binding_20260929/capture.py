"""Capture the original generator's complete actual compiler output only."""
import ast
import builtins
from hashlib import sha256
import json
import marshal
from pathlib import Path
import sys
from types import FunctionType
from unittest.mock import patch

from research.v4.candidates.e1_plain_config_dispatch_20260927 import hooks as parent

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INSTANCE = "E1-01"


def function_clone(function, **bindings):
    namespace = {**function.__globals__, **bindings}
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def capture_parent(output=None):
    """Invoke only original source generator; never main or its callbacks."""
    parent.source_hashes()
    original = parent.compiled
    captures = []
    emitted = []

    def capture_compile(source, filename, mode, *args, **kwargs):
        if not isinstance(source, str) or mode != "exec":
            raise RuntimeError("original own full-module compiler contract changed")
        code = builtins.compile(source, filename, mode, *args, **kwargs)
        captures.append(source)
        emitted.append(code)
        return code

    compiler = function_clone(original, compile=capture_compile)
    module_name = parent.PACKAGE + "._runner_E1_01"
    previous = sys.modules.get(module_name)
    try:
        with patch.object(parent, "compiled", compiler):
            module = parent.runner_module(INSTANCE)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    if len(captures) != 1 or len(emitted) != 1:
        raise RuntimeError("original full-module compiler did not emit exactly once")
    source, whole_code = captures[0], emitted[0]
    main_codes = [value for value in whole_code.co_consts
                  if isinstance(value, type(whole_code)) and value.co_name == "main"]
    if len(main_codes) != 1 or module.main.__code__ is not main_codes[0]:
        raise RuntimeError("returned main is not exact actual emitted CodeType")
    if module.Scene is not module.CONFIGURED.Scene:
        raise RuntimeError("original generated main did not consume CONFIGURED.Scene")
    receipt = {
        "compile_count": 1,
        "original_compiler_CodeType_identity": compiler.__code__ is original.__code__,
        "returned_main_is_actual_emitted_CodeType": module.main.__code__ is main_codes[0],
        "main_globals_is_returned_module": module.main.__globals__ is module.__dict__,
        "Scene_is_CONFIGURED_Scene": module.Scene is module.CONFIGURED.Scene,
        "compiler_scope_restored": parent.compiled is original,
        "module_registry_restored": sys.modules.get(module_name) is previous,
        "parent_module_source_sha256": sha256(source.encode()).hexdigest(),
        "whole_emitted_marshal_sha256": sha256(marshal.dumps(whole_code)).hexdigest(),
        "main_emitted_marshal_sha256": sha256(marshal.dumps(module.main.__code__)).hexdigest(),
        "main_code_filename": module.main.__code__.co_filename,
        "main_code_firstlineno": module.main.__code__.co_firstlineno,
        "main_CodeType_id": id(module.main.__code__),
        "main_globals_id": id(module.main.__globals__),
        "CONFIGURED_id": id(module.CONFIGURED),
        "Scene_module": module.Scene.__module__,
        "Scene_name": module.Scene.__name__,
        "Evaluator_module": module.Evaluator.__module__,
        "Evaluator_name": module.Evaluator.__name__,
        "instruction": module.INSTRUCTION,
        "binding_source": module.CONFIGURED.binding.source,
        "binding_target": module.CONFIGURED.binding.target,
        "original_hashes_reads_source_hashes": "source_hashes" in module.hashes.__code__.co_names,
        "main_called": False,
        "preflight_called": False,
        "Scene_constructed": False,
        "Consumer_constructed": False,
    }
    if not all(receipt[key] for key in (
            "original_compiler_CodeType_identity", "returned_main_is_actual_emitted_CodeType",
            "main_globals_is_returned_module", "Scene_is_CONFIGURED_Scene",
            "compiler_scope_restored", "module_registry_restored")):
        raise RuntimeError("actual original capture/source scope identity differs")
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "parent_module.py.txt").write_text(source)
        (output / "parent_module.ast.txt").write_text(ast.dump(ast.parse(source), include_attributes=True))
        (output / "parent_module.marshal").write_bytes(marshal.dumps(whole_code))
        (output / "parent_main.marshal").write_bytes(marshal.dumps(module.main.__code__))
        (output / "identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    module.capture_receipt = receipt
    return module, source
