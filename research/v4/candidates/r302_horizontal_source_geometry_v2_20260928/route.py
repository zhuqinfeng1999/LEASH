"""Root-authorized-only single own copied route. No Scene on import."""
import inspect
from pathlib import Path
from research.v4.candidates.r302_horizontal_source_geometry_20260928 import copied_route as old
from .helper import bind_contact_helper
H = Path(__file__).resolve().parent
ORIGINAL_H = old.H
CONTACT_SEAM = """ if original_contacts.__code__.co_consts.count('fork_bowl_1_')!=1:raise RuntimeError('Original contact payload prefix drift')
 code=original_contacts.__code__.replace(co_consts=tuple('steel_thermos_1_'if x=='fork_bowl_1_'else x for x in original_contacts.__code__.co_consts))
 contacts=types.FunctionType(code,original_contacts.__globals__)"""
INPUT_SEAM = "geo=json.loads((H/'source_geometry_complete.json').read_text())"

def selected_main():
    source = inspect.getsource(old.main)
    if source.count(CONTACT_SEAM) != 1 or source.count(INPUT_SEAM) != 1:
        raise RuntimeError('Original own copied route source seam drift')
    revised = source.replace(CONTACT_SEAM, ' contacts=bind_contact_helper(original_contacts)')
    revised = revised.replace(INPUT_SEAM, "geo=json.loads((ORIGINAL_H/'source_geometry_complete.json').read_text())")
    ns = dict(old.main.__globals__)
    ns.update(H=H, ORIGINAL_H=ORIGINAL_H, bind_contact_helper=bind_contact_helper)
    exec(compile(revised, str(H/'route.py')+'::one-recursive-contact-fix', 'exec'), ns)
    return ns['main'], source, revised

if __name__ == '__main__':
    selected_main()[0]()
