"""Independent neutral example prompt hook; Root selects and installs it.

Only a selected ordinary UTF-8 English text file is read. No memory, reviewer,
policy, retrieval service, example storage format or runtime source record is
created. The current collector and original planner remain the execution path.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path


NO_EXAMPLE_SENTENCE = "This collection adds no operation examples. "
EXAMPLE_POSITION = "\n\nShared neutral operation examples (development candidate set):\n"
RECOVERY_INSTRUCTIONS = (
    "\n\nCommon bounded manipulation recovery: For each grasp subgoal, use at most "
    "three grasp attempts in this episode and session. If an ordinary grasp fails, "
    "inspect fresh public observations and feedback, relocalize and adjust before "
    "retrying. An approach point is not a final closing position; a completed "
    "motion or CLOSED gripper alone does not prove the object was transported. "
    "Keep all enabled reviews; DENY, BLOCK, unresolved state and correct refusal "
    "are not grasp failures and must not be bypassed. Do not repeat a denied "
    "proposal unchanged. Preserve prior facts and history, including later "
    "instructions and timed events. After three failed grasp attempts report the "
    "actual incomplete result with finish; do not reset the scene or episode."
)

EPISODE_BUDGET_RECOVERY_INSTRUCTIONS = (
    "\n\nCommon manipulation recovery within the episode budget: An ordinary "
    "failed grasp does not require ending after three attempts. In this same "
    "scene and session, use fresh public observations and actual feedback to "
    "relocalize, adjust, and retry while the common primitive, model and wall "
    "budgets permit. Each further attempt must use an updated observation or "
    "a concrete correction; do not repeat an unchanged unsuccessful attempt "
    "without evidence of progress. A pre-grasp standoff is not a closing pose. "
    "Motion completion or CLOSE alone does not establish a grasp; verify "
    "actual object motion. Keep every enabled review. DENY, BLOCK, unresolved "
    "state and correct refusal are not grasp failures and must not be bypassed. "
    "Preserve previous actions, timed events and all round history; do not reset "
    "the scene or episode to erase a failure. Stop with the actual result when "
    "the task is complete, correctly refused, no evidence-supported recovery "
    "remains, or the existing common episode limits are reached."
)


@dataclass(frozen=True)
class PromptBinding:
    condition: str
    text: str | None = None
    path: str | None = None
    version: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    example_count: int | None = None
    recovery_rules: str | None = None
    recovery_rule_version: str | None = None

    def apply(self, system_prompt):
        if self.condition == "no_familiarization" and not self.recovery_rules:
            return system_prompt
        if not isinstance(system_prompt, str):
            raise TypeError("Original planner system_prompt must be text")
        # The current collector sentence is true only for the baseline. Leave
        # all Global Memory/public-tool guidance in the shared prompt intact.
        base = system_prompt
        if self.condition != 'no_familiarization':
            base = base.replace(NO_EXAMPLE_SENTENCE, "", 1) + EXAMPLE_POSITION + self.text
        return base + (self.recovery_rules or '')

    def manifest_fields(self):
        fields = {"condition": self.condition}
        if self.recovery_rules:
            fields['recovery_rule'] = (self.recovery_rule_version
                or 'common-max3-grasp-attempts-same-episode-v1')
        if self.condition == "shared_familiarization":
            fields.update(shared_examples_path=self.path, shared_examples_version=self.version,
                          shared_examples_sha256=self.sha256, shared_examples_bytes=self.bytes)
            if self.example_count is not None:
                fields["shared_examples_count"] = self.example_count
        return fields


def prepare_binding(condition, *, example_file=None, example_version=None,
                    expected_sha256=None, example_count=None):
    """Read the one Root-selected frozen text once, before constructing planner.

    Root records manifest_fields() in the existing run manifest and includes the
    selected file in its finite start/end hash list. Partial candidate text is
    accepted; the helper never reads/fills an eight-slot inventory or infers how
    many examples a prose file contains.
    """
    if condition == "no_familiarization":
        # Do not inspect any shared-example arguments or path in this condition.
        return PromptBinding(condition)
    if condition != "shared_familiarization":
        raise ValueError("Declared no_familiarization/shared_familiarization required")
    if not isinstance(example_version, str) or not example_version.strip():
        raise ValueError("Root-selected example version is required")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("Root-selected frozen example SHA-256 is required")
    if example_count is not None and (type(example_count) is not int or example_count < 1):
        raise ValueError("Actual selected example count must be a positive integer")
    if example_file is None:
        raise FileNotFoundError("Root-selected ordinary example text is missing")
    path = Path(example_file).resolve()
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if not text.strip():
        raise ValueError("Root-selected ordinary example text is empty")
    digest = sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Root-selected frozen example SHA-256 differs")
    return PromptBinding(condition, text, str(path), example_version, digest, len(raw), example_count)


def bind_planner(planner, binding):
    """Change only one fresh planner instance's ordinary solve prompt argument.

    CodexPlanner.solve already takes keyword-only system_prompt/user_message/
    toolkit/max_turns. No SDK configuration, persistent Global Memory or trusted
    base_instructions field is modified. Baseline returns the same planner.
    """
    if binding.condition == "no_familiarization" and not binding.recovery_rules:
        return planner
    original = planner.solve
    def solve(*args, **kwargs):
        if "system_prompt" not in kwargs:
            raise TypeError("Use the original keyword-only system_prompt interface")
        selected = dict(kwargs)
        selected["system_prompt"] = binding.apply(kwargs["system_prompt"])
        return original(*args, **selected)
    planner.solve = solve
    return planner


def wrap_planner_builder(original_builder, binding):
    """Pass as the existing collector's planner_builder argument; starts no IO."""
    if binding.condition == "no_familiarization" and not binding.recovery_rules:
        return original_builder
    def builder(**kwargs):
        return bind_planner(original_builder(**kwargs), binding)
    return builder


@contextmanager
def planner_prompt_scope(binding):
    """Use the same selected text in sync and persistent episode planners.

    The persistent runner already inserts B0_SYSTEM_PROMPT only on its first
    round. Its later round text and retained episode history are untouched.
    The scope changes no policy options, toolkit, reviewer or persistent memory.
    Root enters it around one episode and restores both existing prompt seams.
    """
    if binding.condition == "no_familiarization" and not binding.recovery_rules:
        yield
        return
    from unittest.mock import patch
    from research.v3.bridge import b0
    from research.v4.candidates.multiround_bridge_20260924 import driver
    builder = wrap_planner_builder(b0.build_b0_planner, binding)
    prefix = binding.apply(driver.B0_SYSTEM_PROMPT)
    with patch.object(b0, "build_b0_planner", builder), \
         patch.object(driver, "B0_SYSTEM_PROMPT", prefix):
        yield
