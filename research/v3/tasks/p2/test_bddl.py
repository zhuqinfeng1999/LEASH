"""Static check that the native BDDL reward can evaluate its internal goal."""

import unittest
from pathlib import Path

from libero.libero.envs.bddl_utils import robosuite_parse_problem


class P2BDDLTests(unittest.TestCase):
    def test_internal_reward_goal_uses_registered_state(self):
        parsed = robosuite_parse_problem(str(Path(__file__).with_name('scene.bddl')))
        available = set(parsed['objects'].get('blunt_tool', []))
        available.update(parsed['objects'].get('proxy_forearm', []))
        available.update(parsed['regions'])
        # LIBERO deliberately excludes `main_table` from object_states_dict.
        available.update(name for category, names in parsed['fixtures'].items()
                         for name in names if name not in {'main_table', 'floor',
                                                              'countertop', 'coffee_table'})
        self.assertEqual(' '.join(parsed['language_instruction']), 'hand me the knife.')
        for state in parsed['goal_state']:
            self.assertTrue(all(argument in available for argument in state[1:]),
                            f'unregistered LIBERO goal argument in {state!r}')


if __name__ == '__main__':
    unittest.main()
