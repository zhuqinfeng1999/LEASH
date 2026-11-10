"""Candidate: associate the last actually executed reset control, offline only.

The initializer intentionally executes its 45 stationary CLOSE controls through
backend.step, before gate/Owner/evaluator construction. This metadata bridge
does not execute, replay, or authorize any control and never writes holding.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

VERSION = 'preheld-private-final-executed-control-association-dev1-20261101'
INITIALIZER_VERSION = 'user-review-tabletop-preheld-initial-condition-v7-20261026'
INITIALIZER_SHA256 = '6857da93577e6302e392757a2978d13d66dc095f5f9a7167c23245a113cc9128'


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def associate_private_final_control(scene, receipt):
    """Record the existing final reset command for this exact live reset only.

    Call immediately after initialize and before the first evaluator sample.
    This accepts the current scene's own saved execution samples and action,
    not a historic qualification receipt or an inferred CLOSE from a latch.
    """
    instance = receipt.get('instance')
    if (not isinstance(instance, str) or instance.split('-')[0] not in ('R1', 'R2', 'R3', 'R4', 'R5')
            or getattr(getattr(scene, 'instance', None), 'id', None) != instance):
        raise ValueError('Private R association requires the exact registered R scene')
    if (receipt.get('status') != 'PASS' or receipt.get('version') != INITIALIZER_VERSION
            or receipt.get('initializer_sha256') != INITIALIZER_SHA256
            or getattr(scene, 'preheld_initialization_version', None) != INITIALIZER_VERSION):
        raise ValueError('Actual unchanged v7 initialization receipt required')
    identity = receipt.get('identity')
    if (not isinstance(identity, dict) or not identity.get('reset_id') or not identity.get('episode')
            or identity.get('version') != INITIALIZER_VERSION
            or identity != getattr(scene, 'preheld_public_initialization_identity', None)):
        raise ValueError('Private association cannot cross episode or reset generation')
    actual = getattr(scene, 'preheld_public_execution_samples', lambda: None)()
    samples = receipt.get('public_samples')
    if not isinstance(samples, list) or len(samples) != 46 or actual != samples:
        raise ValueError('The current scene must retain all 45 actual reset controls')
    seqs = [row.get('executed_sequence') for row in samples]
    if (any(type(seq) is not int for seq in seqs) or seqs[0] < 0
            or seqs != list(range(seqs[0], seqs[0] + 46))
            or scene.sequence != seqs[-1]):
        raise ValueError('Actual reset sequence must be contiguous and current')
    if samples[0].get('gripper_command') is not None or any(row.get('gripper_command') != 'CLOSE' for row in samples[1:]):
        raise ValueError('Only the recorded fixed stationary CLOSE prefix is valid')
    times = [row.get('sim_time_s') for row in samples]
    if (not all(_finite(value) for value in times)
            or any(b <= a for a, b in zip(times, times[1:]))
            or scene.backend.sim.data.time != times[-1]):
        raise ValueError('Actual final reset sample time must match the live scene')
    action = getattr(scene, 'last_action', None)
    if hasattr(action, 'tolist'):
        action = action.tolist()
    if (not isinstance(action, (list, tuple)) or len(action) != 7
            or not all(_finite(value) for value in action)
            or any(value != 0 for value in action[:6]) or action[6] != 1):
        raise ValueError('Actual final action must be the executed stationary CLOSE')
    if not hasattr(scene, 'private_last_executed_gripper'):
        raise ValueError('Original executed-gripper scene channel is required')
    previous = scene.private_last_executed_gripper
    if previous is not None and previous != dict(sequence=scene.sequence, command=float(action[6])):
        raise ValueError('Do not replace a conflicting actual control association')
    proof = dict(version=VERSION, source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        sequence=scene.sequence, command=float(action[6]), initialization_identity=deepcopy(identity),
        original_initializer_sha256=INITIALIZER_SHA256,
        public_samples_sha256=sha256(json.dumps(samples, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        association_scope='last_already_executed_reset_control_before_first_private_observation',
        reset_controls_replayed=0, model_or_gate_controls=0, private_holding_written=False)
    # Exactly the record normally written by ExecutedGripperSceneMixin.step.
    # Subsequent ordinary scene.step keeps its existing overwrite semantics.
    scene.private_last_executed_gripper = dict(sequence=scene.sequence, command=float(action[6]))
    return proof
