"""Reuse exact offline first-risk-stop logic with the corrected common scene."""
from types import FunctionType
from research.v4.candidates.r4_06_first_risk_stop_20260926 import scripts as old
from .scene import T1R4CoinDishScene
namespace={**old.namespace,'T1R4CoinDishScene':T1R4CoinDishScene}
first_stop=FunctionType(old.original_with_step_stop.__code__,namespace,name='run_case')
def run_case(seed,case):
 try:return first_stop(seed,case)
 except old.ActualRiskStop as stopped:return stopped.report
