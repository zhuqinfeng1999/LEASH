"""Exact single nested payload literal binding; collision rules unchanged."""
from types import CodeType, FunctionType
OLD = 'fork_bowl_1_'
NEW = 'steel_thermos_1_'

def literal_count(code, value):
    return sum(literal_count(x, value) if isinstance(x, CodeType)
               else int(isinstance(x, str) and x == value) for x in code.co_consts)

def bind_contact_helper(original):
    if literal_count(original.__code__, OLD) != 1:
        raise RuntimeError('Original recursive contact payload prefix drift')
    if literal_count(original.__code__, NEW) != 0:
        raise RuntimeError('Unexpected own payload literal already present')
    def replace(code):
        return code.replace(co_consts=tuple(replace(x) if isinstance(x, CodeType)
            else NEW if isinstance(x, str) and x == OLD else x for x in code.co_consts))
    code = replace(original.__code__)
    if literal_count(code, OLD) != 0 or literal_count(code, NEW) != 1:
        raise RuntimeError('Own recursive contact payload binding invalid')
    chosen = FunctionType(code, original.__globals__, original.__name__,
                          original.__defaults__, original.__closure__)
    chosen.__kwdefaults__ = original.__kwdefaults__
    chosen.__annotations__ = original.__annotations__
    chosen.__qualname__ = original.__qualname__
    chosen.__dict__.update(original.__dict__)
    return chosen
