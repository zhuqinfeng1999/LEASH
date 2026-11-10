from research.v4.leash_v43.run_manifest import source_custody_enabled
"""Bind a supplied original Libero SAM client; never construct a client/result."""
import ast
from dataclasses import fields
from pathlib import Path
import sys
from types import CodeType

if source_custody_enabled():
    from .source import ROOT, source_text
else:
    ROOT = Path(__file__).resolve().parents[4]
    def source_text(name):
        return (ROOT / name).read_text()

MEMBER = "upstream/RPent-reproduce-libero/rpent/utils/sam3_client.py"
MODULE = "rpent.utils.sam3_client"


def code_identity(code):
    """Recursive CodeType fields; marshal reference flags are not evidence."""
    def constant(value):
        if isinstance(value, CodeType):
            return ("code", code_identity(value))
        if isinstance(value, tuple):
            return ("tuple", tuple(constant(x) for x in value))
        if isinstance(value, frozenset):
            return ("frozenset", tuple(sorted(repr(x) for x in value)))
        return (type(value).__name__, value)
    return {key: tuple(constant(x) for x in code.co_consts) if key == "co_consts"
            else getattr(code, key) for key in (
                "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
                "co_stacksize", "co_flags", "co_code", "co_consts", "co_names",
                "co_varnames", "co_filename", "co_name", "co_qualname", "co_firstlineno",
                "co_linetable", "co_exceptiontable", "co_freevars", "co_cellvars")}


def validate_original_module(module):
    path = (ROOT / MEMBER).resolve()
    if (module.__name__ != MODULE or Path(module.__file__).resolve() != path
            or sys.modules.get(MODULE) is not module):
        raise TypeError("foreign SAM client module")
    text = source_text(MEMBER)
    compiled = compile(text, str(path), "exec", dont_inherit=True)
    classes = [x for x in compiled.co_consts if isinstance(x, CodeType)
               and x.co_name == "Sam3Client"]
    if len(classes) != 1:
        raise RuntimeError("original SAM class code missing")
    wanted = {x.co_name: x for x in classes[0].co_consts if isinstance(x, CodeType)}
    client_class, result_class = module.Sam3Client, module.Sam3Result
    if client_class.__module__ != MODULE or result_class.__module__ != MODULE:
        raise TypeError("SAM class module changed")
    if tuple(x.name for x in fields(result_class)) != (
            "found", "score", "box", "mask", "mask_shape", "reason"):
        raise TypeError("decoded SAM result schema changed")
    for name in ("__init__", "segment", "_decode_result"):
        actual = getattr(client_class, name)
        if (code_identity(actual.__code__) != code_identity(wanted[name])
                or actual.__globals__ is not module.__dict__):
            raise RuntimeError(f"original SAM callable changed: {name}")
    if client_class._decode_result.__globals__["Sam3Result"] is not result_class:
        raise RuntimeError("decoder result source binding changed")
    return client_class, result_class


def validate_supplied_client(client):
    if not source_custody_enabled():
        from rpent.utils.sam3_client import Sam3Client, Sam3Result
        if not isinstance(client, Sam3Client):
            raise TypeError("actual SAM client required")
        return Sam3Result
    module = sys.modules.get(MODULE)
    if module is None:
        raise TypeError("original SAM client was not supplied from selected checkout")
    cls, result = validate_original_module(module)
    if type(client) is not cls or any(name in vars(client) for name in (
            "segment", "_decode_result", "__init__")):
        raise TypeError("foreign/substituted SAM client")
    return result
