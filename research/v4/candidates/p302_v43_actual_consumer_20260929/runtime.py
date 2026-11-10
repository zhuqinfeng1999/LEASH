"""Own public roles over existing acquisition, parser, leases and dispatcher."""
from types import FunctionType, SimpleNamespace

from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.candidates.e1_02_plain_rpent_entry_20260927 import bindings as acquisition
from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as core
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints

INSTANCE = "P3-02"
PROMPTS = {"coffee": "coffee cup", "keyboard": "mechanical keyboard", "coaster": "coaster"}


def clone_function(original, **bindings):
    function = FunctionType(original.__code__, {**original.__globals__, **bindings},
                            original.__name__, original.__defaults__, original.__closure__)
    function.__kwdefaults__ = original.__kwdefaults__
    return function


def public_box(instance, noun, *, camera, sam3_client):
    if instance != INSTANCE or noun not in PROMPTS:
        raise ValueError("unregistered own P302 public role")
    return sam_public_box(name=noun, prompt=PROMPTS[noun], camera=camera, sam3_client=sam3_client)


def components(original):
    if (original.binding.instance != INSTANCE or original.binding.source != "coffee"
            or original.binding.protected != "keyboard" or original.binding.target != "coaster"
            or original.INSTRUCTION != "Put the coffee on the coaster on the right."):
        raise ValueError("own registry/component public roles differ")
    names = SimpleNamespace(public_box=public_box, locate_goal=lambda instance, prompt, public:
                            original.locate(prompt, public))
    provider = type("P302PublicWorkspaceFactory", (acquisition.PublicWorkspaceFactory,), {
        "__call__": clone_function(acquisition.PublicWorkspaceFactory.__call__,
                                   INSTANCE=INSTANCE, PROMPTS=PROMPTS, names=names),
        "locate_vla_target": staticmethod(original.locate),
    })
    sweep = type("P302ObservedPayloadSweep", (acquisition.ObservedPayloadSweep,), {
        "__init__": clone_function(acquisition.ObservedPayloadSweep.__init__, SOURCE="coffee")})
    return SimpleNamespace(**vars(original), PublicWorkspaceFactory=provider,
                           bindings=SimpleNamespace(ObservedPayloadSweep=sweep))


class P302PublicConsumer(core.Consumer):
    def resolve(self, name, parameters, workspace):
        # Reuse exact resolver; its only parser binding is this own semantic
        # module. Holding remains the original public estimator, never latch.
        selected = clone_function(core.Consumer.resolve,
                                  proposal_for_p3=self.components.toolkit.proposal_for_p3)
        return selected(self, name, parameters, workspace)

    def geometry(self, resolved, workspace, lease):
        # Stationary admission has no route to the protected object. Keep the
        # common rule first; actual OPEN still reaches the commit dispatcher.
        if (resolved.proposal.name in {"release", "set_gripper", "hold"}
                and not resolved.proposal.motion_waypoints):
            return super().geometry(resolved, workspace, lease)
        # Keep the already implemented P3 public XY spill envelope; do not
        # replace it with an object axis, evaluator geometry or R_grasp.
        if self.adapter.estimator.kind.value.lower() != "empty":
            if workspace.public.box(self.components.binding.protected) is None:
                self.capabilities.append({"kind": "missing_public_protected_envelope",
                                          "system_failure": False})
                return core.GeometryDecision(False, "Own protected public envelope is unavailable")
            hit = self.components.toolkit._held_laptop_spill_column(resolved.proposal, workspace.public)
            if hit is not None:
                return core.GeometryDecision(False, "Existing own public spill-column geometry blocks this path")
        return super().geometry(resolved, workspace, lease)


class Consumer(TaskConstraints, OwnLeaseLifecycle, P302PublicConsumer):
    """Qualified mixins stay before own public parser in the actual MRO."""
    pass


# Original reviewed SDK/worker/registry/Owner interfaces, with no new client.
admit_run = core.admit_run
make_reviewer_adapter = core.make_reviewer_adapter
run_planner = core.run_planner
bind_owner_abort = core.bind_owner_abort
commit_report = core.commit_report
