"""I1 native scene: printed system-impersonation note is untrusted image data."""
from .core import InjectionScene

class I1Scene(InjectionScene):
    def __init__(self, seed=0, size=256):
        super().__init__('I1',seed=seed,size=size)
