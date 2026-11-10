"""S101 resolver/provider choices over the unchanged round Consumer/outlet."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
from dataclasses import asdict, replace
import json
from types import FunctionType, SimpleNamespace

from research.v4.candidates.s101_v43_round_consumer_20260929.runtime import S101RoundConsumer
from research.v4.candidates.s101_v43_public_provider_20260929 import public as own_public
from research.v4.candidates.e1_02_plain_rpent_entry_20260927.bindings import ObservedPayloadSweep as ExistingSweep
from research.v4.leash_v43.method_inputs import validate_review_response, InvalidReviewResponse
from research.v4.leash_v43.tool_boundary import GeometryDecision, ResolvedPrimitive


def gate1_constraints(checked):
    """Retain semantic task restrictions verbatim until gate 2 binds entities."""
    from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import merge_texts
    lease = checked.get("lease") or {}
    extra = list(lease.get("commit_conditions", lease.get("conditions", ())))
    if lease.get("no_go_regions"):
        extra.append("Gate 1 no-go regions: " + json.dumps(lease["no_go_regions"], ensure_ascii=False))
    return merge_texts(checked.get("constraints", ()), lease.get("task_constraints", ()), extra)


def own_payload_sweep_class():
    """Same existing full AABB-corner/EEF code; only public source noun is pan.

    No E2 landing predicate or numerical object length is consumed. The reused
    helper's original exception wording is retained; this wrapper prechecks the
    public pan box and never treats such a message as a semantic classifier.
    """
    original = ExistingSweep.__init__
    globals_ = dict(original.__globals__, SOURCE="pan")
    init = FunctionType(original.__code__, globals_, original.__name__,
                        original.__defaults__, original.__closure__)
    init.__kwdefaults__ = original.__kwdefaults__
    if source_custody_enabled() and (init.__code__ is not original.__code__ or init.__globals__["SOURCE"] != "pan"):
        raise RuntimeError("existing public corner/EEF sweep binding changed")
    return type("S101ObservedPayloadSweep", (ExistingSweep,), {"__init__": init})


class S101OwnPublicConsumer(S101RoundConsumer):
    def task_review(self, **kwargs):
        checked = super().task_review(**kwargs)
        if checked["result"] == "PROCEED" and checked.get("lease"):
            # Gate 1's semantic conditions must survive into the first bound
            # action lease. They are review evidence, never an action permit.
            self._task_constraints = gate1_constraints(checked)
        return checked

    def __init__(self, *args, components=None, **kwargs):
        if "provider_factory" in kwargs:
            raise ValueError("this version selects its own reviewed S101 provider")
        selected = own_public.selected_components() if components is None else components
        if source_custody_enabled() and (selected.original is not own_public.own.components("S1-01")
                or selected.PublicWorkspaceFactory.__call__.__globals__.get("_Client") is not own_public._Client
                or selected.s1_public_proposal.__globals__.get("_unknown_geometry") is not own_public._unknown_geometry
                or selected.original.binding.instance != "S1-01"
                or selected.original.binding.noun != "pan"
                or tuple(selected.original.instructions) != (
                    "Move the pan onto the wooden board.", "Put the pan in my hands.")):
            raise ValueError("exact own S101 definitions required")
        self.own_public = selected
        self.ObservedPayloadSweep = own_payload_sweep_class()
        self.role_logs = []
        def provider_factory(actual_components, scene, adapter, sam_client):
            if actual_components is not selected:
                raise RuntimeError("common constructor selected another entity configuration")
            return own_public.make_provider(scene=scene, adapter=adapter, sam3_client=sam_client,
                forbidden_entities=self.current_public_entities, components=selected)
        super().__init__(*args, components=selected, provider_factory=provider_factory, **kwargs)

    def current_public_entities(self):
        """Current validated structural fields only, never reason/text parsing."""
        if not self.boundary.round_open or not self.boundary.task_approved:
            return ()
        current = self.boundary.lease
        if current is None or not self.lease_is_current(current):
            # The first action has no grounded lease. Acquire the existing
            # public query vocabulary for gate 2 to bind; gate 1's free text
            # must not be interpreted as mandatory SAM query identifiers.
            return tuple(sorted(self.own_public.prompts))
        leases = [current]
        names = set()
        for lease in leases:
            values = [*lease.get("entities", ())]
            # Explicit reviewer-authored geometric boxes are not object-role
            # labels and require no invented SAM prompt. Named regions do.
            values.extend(r for r in lease.get("no_go_regions", ()) if isinstance(r, str))
            for value in values:
                try:
                    names.update(self.own_public.original.grounding.canonical_entities(value))
                except ValueError as exc:
                    # This value came from the reviewer, before any SAM query.
                    # An unsupported role is an invalid reply, not a camera fault.
                    raise InvalidReviewResponse(str(exc)) from exc
        if not names <= set(self.own_public.prompts):
            raise InvalidReviewResponse("validated own public role has no original S1 query")
        result = tuple(sorted(names))
        self.role_logs.append({"round_id": self.boundary.round_id, "binding_version": self.binding_version,
                               "public_names": result, "source": "validated_current_structural_lease_fields"})
        return result

    def resolve(self, name, parameters, workspace):
        public = workspace.public
        kind = self.adapter.estimator.kind.value.lower()
        box = public.box("pan")
        missing = kind != "empty" and box is None
        parser_public = public
        if kind == "held" and missing:
            # Preserve HELD in actual proprio/reviewer/registry. Only proposal
            # shape is conservative UNKNOWN when the visible payload is absent.
            parser_public = replace(public, evidence=SimpleNamespace(
                robot=SimpleNamespace(hold_estimate="unknown")))
        proposal = self.own_public.s1_public_proposal(
            name, parameters, parser_public, self.provider.locate_vla_target)
        if kind != "empty" and box is not None:
            sweep = self.ObservedPayloadSweep(public)
            proposal = replace(proposal, operated_entity="pan",
                gripper_radius_m=sweep.gripper_radius_m, held_object_radius_m=sweep.held_extra_radius_m)
        proposal = replace(proposal, geometry_ambiguous=proposal.geometry_ambiguous or missing or kind == "unknown")
        self._public_phase = None  # Own S1 algorithm has no explicit phase field.
        if self.lease_receipt is not None and self._lease_phase is not None:
            self._expire("existing S1 public proposal phase became unknown")
        self._version(workspace)
        self.proposals.append({"name": name, "arguments": dict(parameters), "proposal": asdict(proposal),
            "source": "own_s1_public_proposal_and_existing_full_corner_sweep",
            "holding_source": "executed_records_and_public_proprio"})
        return ResolvedPrimitive(workspace.version, proposal,
            entities=tuple(x for x in (proposal.operated_entity,) if x), geometry_unknown=proposal.geometry_ambiguous)

    def geometry(self, resolved, workspace, lease):
        proposal = resolved.proposal
        # Missing payload is relevant to motion, including a VLA target or a
        # held rotation's EEF envelope. Unrelated nomotion hold/release/closure
        # still uses existing admission and actual registry, never semantic EMPTY.
        if (self.adapter.estimator.kind.value.lower() != "empty"
                and workspace.public.box("pan") is None
                and (proposal.motion_waypoints or proposal.name == "vla_act")):
            self.capabilities.append({"kind": "missing_own_public_pan_envelope", "system_failure": False})
            return GeometryDecision(False, "Current public pan payload envelope is unavailable")
        return super().geometry(resolved, workspace, lease)
