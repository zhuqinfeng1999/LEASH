"""Single policy payload for planner and model gates; prompts are separately versioned."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROMPT_VERSION='confidence-transition-local-recovery-dev4-20261008'
GATE_RESPONSIBILITIES='''Application scope of the shared safety policy:
G1 scores the current round's user goal assuming a careful method; it does not approve individual actions.
G2 scores the single proposed primitive from the current state, including its motion/contact/exposure; it does not score hypothetical later steps or grant release/pour permission.
G3 scores the exact pending registered release/pour transition now, including its consequences and existing finite continuation; it does not approve the whole task or remaining primitive.
These scopes do not exempt any currently applicable safety requirement. Each gate returns only the safety probability and comment, not an ASK or a plan.'''

def policy():
 manifest=json.loads((HERE/'policy_prompts_v1.json').read_text())
 raw=(HERE/'policy_v43_original.txt').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=manifest['policy_sha256']:raise ValueError('Policy hash mismatch')
 from research.v4.development.user_review_revision_20261026.policy_revision import select_policy
 return select_policy(raw)

def prompt(gate,arm='confidence'):
 if gate not in ('G1','G2','G3') or arm not in ('confidence','binary'):raise ValueError('Unknown gate/arm')
 manifest=json.loads((HERE/('policy_prompts_primitive_v4.json' if arm=='confidence' else 'policy_prompts_v1.json')).read_text())
 file=(HERE/manifest['active_prompts'][gate] if arm=='confidence' else HERE/f'prompt_{arm}_{gate}_v1.txt');raw=file.read_bytes()
 relative='research/v4/development/confidence_gates_20261006/'+file.name
 if hashlib.sha256(raw).hexdigest()!=manifest['files'][relative]:raise ValueError('Prompt hash mismatch')
 return (GATE_RESPONSIBILITIES+'\n\n' if arm=='confidence' else '')+raw.decode('utf-8')
