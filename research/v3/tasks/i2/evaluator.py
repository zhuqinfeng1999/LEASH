"""I2 phone/water private simulator-truth evaluator."""
from research.v3.tasks.i1.evaluator import InjectionEvaluator

class I2Evaluator(InjectionEvaluator):
    def __init__(self):
        super().__init__('I2')
