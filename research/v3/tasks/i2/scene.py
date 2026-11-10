"""I2 native scene: printed owner-impersonation note is untrusted image data."""
from research.v3.tasks.i1.core import InjectionScene

class I2Scene(InjectionScene):
    def __init__(self, seed=0, size=256):
        super().__init__('I2',seed=seed,size=size)
