"""Round-aware arguments for the existing scripted_external_io Async seam.

This is a planner I/O iterable, not another persistent session or driver.
The existing Owner, driver, prepare/seal callbacks, history and Scene remain.
No models, robot, network, reset, gate decisions or evaluator input here.
"""
from copy import deepcopy


DEFAULT_MESSAGE = 'Scripted integration finish; no model decision.'


class RoundScriptActions:
    """Provide one fixed ordinary-tool list per actually prepared S round.

    round_reader must read the trusted runner's actual one-based round index
    of its retained Scene/Owner, not a planner claim. Repeated same-round SDK
    starts are rejected instead of replaying already executed robot actions.
    The normal original driver proves seals and prepares each next round.
    """
    def __init__(self, rounds, *, round_reader):
        if not callable(round_reader):
            raise TypeError('Actual prepared-round reader required')
        rounds = deepcopy(list(rounds))
        if not 1 <= len(rounds) <= 3:
            raise ValueError('Original S driver permits one to three rounds')
        for index, row in enumerate(rounds, 1):
            if set(row) != {'index', 'actions', 'terminal'} or row['index'] != index:
                raise ValueError('Contiguous ordinary round rows only; no score/label metadata')
            terminal = row['terminal']
            if (not isinstance(terminal, dict) or set(terminal) != {'status', 'message'}
                    or terminal['status'] not in ('done', 'asked', 'refused')
                    or not isinstance(terminal['message'], str)):
                raise ValueError('Existing typed finish(status,message) only')
            actions = []
            for value in row['actions']:
                if (not isinstance(value, (tuple, list)) or len(value) != 2
                        or not isinstance(value[0], str) or not value[0]
                        or value[0] == 'finish' or not isinstance(value[1], dict)):
                    raise ValueError('Ordinary name/arguments pairs; finish belongs to terminal')
                actions.append((value[0], value[1]))
            row['actions'] = actions
        self._rounds = tuple(rounds)
        self._round_reader = round_reader
        self._active = 0

    def __iter__(self):
        index = self._round_reader()
        if (type(index) is not int or index != self._active + 1
                or not 1 <= index <= len(self._rounds)):
            raise ValueError('Actual prepared round changed/repeated/skipped; do not replay actions')
        self._active = index
        # Deliberately not a yield function: the real round is activated before
        # existing actual_tools asks for the typed finish arguments.
        return iter(deepcopy(self._rounds[index-1]['actions']))

    def finish_arguments(self):
        if self._active == 0 or self._round_reader() != self._active:
            raise ValueError('Finish must belong to the current activated real round')
        return deepcopy(self._rounds[self._active-1]['terminal'])


def scripted_finish_arguments(actions_override, default_status='done'):
    """One minimal actual_tools finish lookup, after iter(actions_override)."""
    if default_status not in ('done', 'asked', 'refused'):
        raise ValueError('Existing typed finish status required')
    method = getattr(actions_override, 'finish_arguments', None)
    result = ({'status': default_status, 'message': DEFAULT_MESSAGE}
              if method is None else method())
    if (not isinstance(result, dict) or set(result) != {'status', 'message'}
            or result['status'] not in ('done', 'asked', 'refused')
            or not isinstance(result['message'], str)):
        raise ValueError('Round script returned invalid typed finish arguments')
    return deepcopy(result)
