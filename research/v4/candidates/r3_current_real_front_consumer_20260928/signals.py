"""Typed perception causes only; reasons/instance hazard labels are not classifiers."""
from research.v4.candidates.e1_02_plain_rpent_v2_20260927.signals import initial_missing_has_other_fault
from research.v4.candidates.e1_unknown_tilt_typed_20260927.outlet import commit_infrastructure_failure


class NoPublicMask(Exception):
    """Only a successfully decoded real Sam3Result(found=False) issues this."""


class MissingPublicEvidence(Exception):
    def __init__(self, missing, perception, *, allowed_names):
        if (not isinstance(allowed_names, frozenset) or not missing or
            any(not isinstance(n, str) or n not in allowed_names for n in missing)):
            raise ValueError('Missing public names must belong to configured public roles')
        self.missing = tuple(sorted(set(missing)))
        self.perception = tuple(perception)
        super().__init__('Valid initial public capture had no mask for: '+', '.join(self.missing))

    def marker(self):
        return dict(kind='typed_initial_public_masks_missing', version=1,
                    boundary='initial_public_preflight', valid_capture=True,
                    missing_names=list(self.missing), perception=list(self.perception))


def infrastructure_fault(report):
    rows = (report.get('commit_gate') or {}).get('decisions', [])
    commit_fault = commit_infrastructure_failure(rows)
    action_fault = any(x.get('system_failure') for x in (report.get('action_gate') or {}).get('logs', []))
    if report.get('initial_public_evidence_missing'):
        return initial_missing_has_other_fault(report, action_system_fault=action_fault,
                                               commit_system_fault=commit_fault) or bool(
            (report.get('task_gate') or {}).get('system_failure'))
    return bool(report.get('error') or report.get('observation_fault') or
                report.get('source_changes_after') or action_fault or commit_fault)
