"""Per-round lazy stages for the existing confidence scripted planner I/O.

Root's existing scripted.main selects this adapter only for case['rounds'].
It reuses RoundScriptActions activation/typed finish and scripted.actions tail
handling. It starts no session/driver/Owner/model/physics on import or creation.
"""
from copy import deepcopy
from pathlib import Path

from .s_round_script_io import DEFAULT_MESSAGE, RoundScriptActions


class RoundStageActions(RoundScriptActions):
    """One lazy existing actions generator for each actual adapter.round_id."""
    def __init__(self, case, holder, journal, output, *, actions_factory=None):
        if not isinstance(case, dict) or not str(case.get('instance', '')).startswith(('S1-', 'S2-', 'S3-')):
            raise ValueError('Registered S case required for round-stage scripted I/O')
        if case.get('script_ready') is False:
            raise ValueError('Missing collected parameters: case is not ready')
        rows = case.get('rounds')
        if not isinstance(rows, (tuple, list)) or not 1 <= len(rows) <= 3:
            raise ValueError('One to three explicitly registered S rounds required')
        if actions_factory is not None and not callable(actions_factory):
            raise TypeError('Existing lazy actions function must be callable')
        base = deepcopy(case)
        base.pop('rounds')
        # The old single-round fields cannot accidentally shadow a later
        # round's list or typed finish; no global generator is reused.
        base.pop('stages', None)
        base.pop('required_terminal', None)
        round_cases, round_contract = [], []
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict) or row.get('index') != index:
                raise ValueError('S rounds must retain contiguous actual instruction order')
            if row.get('script_ready') is False:
                raise ValueError('Missing collected parameters: round is not ready')
            stages = row.get('stages')
            if not isinstance(stages, (tuple, list)):
                raise ValueError('Collected ordinary stages must exist; do not invent missing targets')
            for stage in stages:
                if (not isinstance(stage, dict) or not isinstance(stage.get('tool'), str)
                        or not stage['tool'] or stage['tool'] == 'finish'
                        or not isinstance(stage.get('arguments'), dict)):
                    raise ValueError('Only ordinary tool stages; typed finish is separate')
                if (stage['arguments'].get('xyz', ()) is None
                        and 'translation_from_actual_public_p0_m' not in stage
                        and 'xyz_from_current_public_eef' not in stage):
                    raise ValueError('Uncollected xyz is not an executable target')
            terminal = deepcopy(row.get('required_terminal') or {'status':'done'})
            if set(terminal) - {'status', 'message'}:
                raise ValueError('Only typed terminal status/message, no score or label input')
            terminal.setdefault('message', DEFAULT_MESSAGE)
            current = deepcopy(base)
            current.update(deepcopy(row))
            current['stages'] = deepcopy(list(stages))
            current['required_terminal'] = terminal
            round_cases.append(current)
            round_contract.append(dict(index=index, actions=[], terminal=terminal))
        self._round_cases = tuple(round_cases)
        self._holder, self._journal, self._output = holder, journal, Path(output)
        self._actions_factory = actions_factory
        def actual_round():
            # Only trusted runner administrative identity; no physics/object
            # truth, holding reset, evaluator label or guessed round counter.
            return holder['stack'].adapter.round_id
        super().__init__(round_contract, round_reader=actual_round)

    def __iter__(self):
        super().__iter__()  # Activates once, before finish lookup; no stage consumed.
        factory = self._actions_factory
        if factory is None:
            from .scripted import actions
            factory = actions
        current = deepcopy(self._round_cases[self._active-1])
        # Calling the existing generator does not run its body. Each relative
        # target/return/tail/private-risk check is still evaluated only when
        # the original actual_tools loop requests that next ordinary stage.
        return iter(factory(current, self._holder, self._journal, self._output))
