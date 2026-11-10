"""Unchanged T3 predicate, using the real ice-pick endpoints; private only."""
import inspect,textwrap
from research.v4.candidates.p2_v42_t3_20260925 import evaluator as base
for k,v in vars(base).items():
 if not k.startswith('__'):globals().setdefault(k,v)
TIP_LOCAL_M=np.array([.114,0.,0.]);HANDLE_END_LOCAL_M=np.array([-.104,0.,0.])
exec(compile(textwrap.dedent(inspect.getsource(base.P2Evaluator.observe)),__file__,'exec'),globals())
class P2IcePickEvaluator(base.P2Evaluator):pass
P2IcePickEvaluator.observe=observe
