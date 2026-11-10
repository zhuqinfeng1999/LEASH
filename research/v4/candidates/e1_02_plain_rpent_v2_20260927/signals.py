"""Offline run-cause classification, not reviewer policy or hidden truth."""
class MissingPublicEvidence(Exception):
    def __init__(self, missing, perception):
        if not missing or any(x not in {'cleaver', 'cutting board', 'person', 'hand'} for x in missing):
            raise ValueError('invalid initial public missing-entity cause')
        self.missing = tuple(sorted(set(missing)))
        self.perception = tuple(perception)
        super().__init__('Valid initial public capture had no mask for: '+', '.join(self.missing))

    def marker(self):
        return {'kind': 'typed_initial_public_masks_missing', 'version': 1,
                'boundary': 'initial_public_preflight', 'valid_capture': True,
                'missing_names': list(self.missing), 'perception': list(self.perception)}


def initial_missing_has_other_fault(report, *, action_system_fault, commit_system_fault):
    marker = report.get('initial_public_evidence_missing')
    if not isinstance(marker, dict) or marker.get('kind') != 'typed_initial_public_masks_missing' or marker.get('valid_capture') is not True:
        return True
    if marker.get('boundary') != 'initial_public_preflight' or not marker.get('missing_names'):
        return True
    # Only a typed exception raised before primitive dispatch may bypass the
    # generic exception classification. Any separately recorded fault wins.
    return bool(report.get('primitive_calls') or report.get('observation_fault') or
        report.get('saved_rgbd_validation_error') or report.get('source_changes_after') or
        action_system_fault or commit_system_fault or report.get('planner_error') or
        report.get('sdk_fault'))
