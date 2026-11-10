"""R4 public RGB-D and frozen VLA input adapter.

R4 uses the same camera protocol and controller interface as R2; the scene
supplies a different object/target configuration and the common grasp assist.
"""

from research.v4.accelerated.r2_adapter import R2V42Adapter


class R4V42Adapter(R2V42Adapter):
    pass
