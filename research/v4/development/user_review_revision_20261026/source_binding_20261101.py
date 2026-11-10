"""Explicit I-task reset source binding around the unchanged v7 initializer.

Only the twelve registered I instances have the authored ketchup-to-basket
goal. This temporary name selects that existing assist entry during reset; it
does not edit objects, physics, assist entries, scoring or later online state.
"""
from hashlib import sha256
import json
from pathlib import Path

from research.v4.development.user_review_revision_20261026 import initialization as _base

VERSION = 'i-goal-ketchup-reset-source-binding-dev1-20261101'
OBJECT_NAME = 'ketchup_1'
INITIALIZER_SHA256 = '6857da93577e6302e392757a2978d13d66dc095f5f9a7167c23245a113cc9128'
INSTANCES = frozenset(f'I{family}-{number:02}' for family in range(1,5) for number in range(1,4))
replay_public_samples = _base.replay_public_samples
_MISSING = object()


class SourceBindingError(ValueError):
    pass


def initialize(scene, *, output, instance, seed):
    if instance not in INSTANCES:
        raise SourceBindingError('Ketchup reset binding is limited to the twelve registered I instances')
    if sha256(Path(_base.__file__).read_bytes()).hexdigest() != INITIALIZER_SHA256:
        raise SourceBindingError('Original v7 initializer source changed')
    objects = scene.backend.objects_dict
    if OBJECT_NAME not in objects:
        raise SourceBindingError('Authored ketchup_1 object is missing; never choose an injected object')
    body = objects[OBJECT_NAME].root_body
    entries = scene.grasp_assist._entries
    matches = [entry for entry in entries if entry[0].object_body == body]
    if len(matches) != 1:
        raise SourceBindingError('Authored ketchup body needs exactly one original assist entry')
    original = getattr(scene, 'item_name', _MISSING)
    proof = dict(version=VERSION,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        object_name=OBJECT_NAME,source_body=body,exact_authored_assist_matches=1,
        temporary_reset_only=True,restored=False)
    receipt = None
    try:
        scene.item_name = OBJECT_NAME
        receipt = _base.initialize(scene,output=output,instance=instance,seed=seed)
        index = scene.grasp_assist._held_index
        actual_body = entries[index][0].object_body if type(index) is int and 0<=index<len(entries) else None
        proof['actual_final_held_body'] = actual_body
        proof['only_authored_goal_source_held'] = (receipt.get('source_body')==body and actual_body==body)
        if not proof['only_authored_goal_source_held']:
            receipt['status'] = 'FAILED'
            receipt['error'] = 'SourceBindingError: Final actual held object differs from authored ketchup'
            raise SourceBindingError('Final actual held object differs from authored ketchup; refuse reset qualification')
        return receipt
    finally:
        if original is _MISSING:
            delattr(scene,'item_name')
        else:
            scene.item_name = original
        restored = getattr(scene,'item_name',_MISSING)
        proof['restored'] = restored is original
        path = Path(output)/'receipt.json'
        if receipt is None and path.is_file():
            receipt = json.loads(path.read_bytes())
        if receipt is not None:
            # Original initializer SHA/status/physical samples remain actual.
            # This record merely adds the separately versioned reset binding.
            receipt['source_binding'] = proof
            path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
