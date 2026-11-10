"""Exact R304 prefix and existing actual action/contact logger, CPU authorized."""
import inspect,hashlib,types
from pathlib import Path
from research.v4.candidates.r3_04_foil_sandwich_20260926 import scripts as own
from research.v4.candidates.r3_04_foil_sandwich_20260926.scene import T1R3FoilSandwichScene
from research.v4.candidates.r3_05_oven_only_prefix_20260928 import root_prefix as logger
from .scene import TallFrontFoilScene
H=Path(__file__).resolve().parent
SEAM="            move([cavity[0], -.12, 1.23], 'front_staging', grip=1.)"
def compiled_prefix(Scene):
 src=inspect.getsource(own.run_case)
 if src.count(SEAM)!=1:raise ValueError('Exact own foil afterlift seam drift')
 at=src.index(SEAM);before=src[:at]
 src=src[:at]+"            details.update(prefix_only=True,attached=attached,lift_m=lift,held_after_lift=bool(attached and lift>=.020),passed=bool(attached and lift>=.020),terminal_status='prefix_diagnostic_complete',evaluator=evaluator.finish('done'))\n            return details\n"+src[at:]
 assert src[:at]==before
 ns=dict(own.run_case.__globals__);ns['R3TabletopFacade']=Scene
 exec(compile(src,str(H/'prefix.py')+'::exact-own-R304-prefix','exec'),ns)
 return ns['run_case'],src,hashlib.sha256(before.encode()).hexdigest()
def selected_contacts():
 src=inspect.getsource(logger.contact_rows)
 assert src.count("'fork_bowl_1_'")==1
 src=src.replace("'fork_bowl_1_'","'foil_sandwich_1_'")
 ns=dict(logger.contact_rows.__globals__);exec(compile(src,str(H/'prefix.py')+'::own-foil-contact-identity','exec'),ns)
 return ns['contact_rows']
def selected_main():
 ns=dict(logger.main.__globals__);ns.update(H=H,T1R3ForkBowlScene=T1R3FoilSandwichScene,OvenOnlyPrefixScene=TallFrontFoilScene,compiled_prefix=compiled_prefix,contact_rows=selected_contacts())
 return types.FunctionType(logger.main.__code__,ns,logger.main.__name__,logger.main.__defaults__,logger.main.__closure__)
if __name__=='__main__':selected_main()()
