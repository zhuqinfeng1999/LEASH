"""Retain literal no-go geometry without treating its label as an image query.

Only the entity-source seam changes. Original recovery admission, geometry,
park/HOLD controls, budgets, holding estimator and sealed lifecycle remain in
the immutable parent. No current main/core has adopted this candidate.
"""
from types import FunctionType

from research.v4.candidates.s101_v43_current_core_consumption_v2_20260929 import runtime as parent
from research.v4.candidates.s101_v43_own_public_consumer_geometry_v2_20260929 import runtime as own
from research.v4.leash_v43.method_inputs import InvalidReviewResponse


class RecoveryScope(parent.RecoveryScope):
    def current_public_entities(self):
        # Use the original normal-role callback, not the draft recovery
        # callback that turns a literal region label into an entity name.
        names = (set(self.own_public.prompts) if self._recovery_context is not None
                 else set(own.S101OwnPublicConsumer.current_public_entities(self)))
        # Gate 1 text is retained by TaskConstraints, not reinterpreted here.
        # Recovery uses the same current action binding/public vocabulary.
        if not names <= set(self.own_public.prompts):
            raise InvalidReviewResponse("validated recovery public role has no original S1 query")
        return tuple(sorted(names))


class S101CoreConsumer(RecoveryScope, own.S101OwnPublicConsumer):
    pass


def core_context(parent_module, *, policy, calibration, source_hashes):
    """Bind the exact original constructor call to this Consumer in isolation.

    This metadata object is not installed in a selected main. It cannot remove
    the separate compiler/source admission STOP or authorize any launch.
    """
    original = parent.CoreBindings.make_consumer
    namespace = dict(original.__globals__, S101CoreConsumer=S101CoreConsumer)
    method = FunctionType(original.__code__, namespace, original.__name__,
                          original.__defaults__, original.__closure__)
    method.__kwdefaults__ = original.__kwdefaults__
    selected = type("S101LiteralRegionCoreBindings", (parent.CoreBindings,),
                    {"make_consumer": method})
    return selected(parent_module, policy=policy, calibration=calibration,
                    source_hashes=source_hashes)
