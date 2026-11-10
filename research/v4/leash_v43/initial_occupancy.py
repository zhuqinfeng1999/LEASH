"""Episode-local public initialization; no env, object or weld access."""
from collections.abc import Mapping

VERSION = 'known-empty-start-dev1-20261005'


def empty_reset_fact(*, episode, reset_id, initial_step, fresh, recipe_qualified):
    if not fresh or not recipe_qualified:
        return None
    if not episode or not reset_id or initial_step != 0:
        raise ValueError('fresh empty initialization requires episode/reset/step zero')
    return dict(initial_occupancy='EMPTY', source='verified_empty_reset',
                protocol=VERSION, episode=episode, reset_id=reset_id, initial_step=0)


def validate_initial_fact(value, *, episode=None):
    keys = {'initial_occupancy', 'source', 'protocol', 'episode', 'reset_id', 'initial_step'}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError('invalid public initialization fact')
    if (value['initial_occupancy'] != 'EMPTY' or value['source'] != 'verified_empty_reset'
            or value['protocol'] != VERSION or type(value['initial_step']) is not int
            or value['initial_step'] != 0 or not isinstance(value['episode'], str)
            or not value['episode'] or not isinstance(value['reset_id'], str)
            or not value['reset_id'] or episode is not None and value['episode'] != episode):
        raise ValueError('wrong initialization source or episode scope')
    return dict(value)
