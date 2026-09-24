import unittest

import numpy as np

from rlcard.games.tractors.env.botzone import tractorGame
from rlcard.games.tractors.env.env import Env
from rlcard.games.tractors.model.play_model import compute_strategy_targets


class TestTractorsStrategyLabels(unittest.TestCase):
    def play_trick(self, hands, banker=0, score=0, level='2'):
        env = Env()
        game = env._env
        game.setMajor('s', level)
        game.player_level = [level] * 4
        game.globalInfo = {
            'stage': 'play', 'level': level, 'game_score': score,
            'banking': {'banker': banker, 'banker_last': (banker + 1) % 4, 'major': 's'},
            'publiccard': [], 'history': [[], [], 0, 0], 'playerpos': 0,
        }
        game.player_hand_cards = [list(hand) for hand in hands]
        for seat, hand in enumerate(hands):
            tractorGame.step(game, [seat, [hand[0]]])
        self.assertEqual(game.getErrorCode(), 0)
        return env, env._get_step_meta()

    def label(self, meta, role):
        label = compute_strategy_targets(meta, role)
        self.assertEqual(label.dtype, np.float32)
        self.assertEqual(float(label.sum()), 1.)
        return int(label.argmax())

    def test_terminal_roles_survive_both_kinds_of_banker_change(self):
        for banker in range(4):
            for score, next_offset in ((0, 2), (80, 1)):
                with self.subTest(banker=banker, score=score):
                    env, meta = self.play_trick([[9], [13], [17], [21]], banker, score)
                    game = env._env
                    self.assertEqual(game.getBanker(), (banker + next_offset) % 4)
                    winner = game._role_of(3, banker)
                    lead = game._role_of(0, banker)
                    self.assertEqual(meta['winner_role'], winner)
                    self.assertEqual(env._get_last_winner_role(), winner)
                    self.assertEqual(env._get_round_lead_role(), lead)
                    self.assertEqual(meta['roles'][lead]['is_lead'], 1.)
                    self.assertEqual(meta['roles'][winner]['won'], 1.)
                    self.assertEqual(self.label(meta, winner), 5)
                    for seat in range(4):
                        role = game._role_of(seat, banker)
                        if seat % 2 == 3 % 2:
                            self.assertEqual(self.label(meta, role), 5)
                        else:
                            self.assertNotEqual(self.label(meta, role), 5)

    def test_finalend_also_freezes_banker_and_terminal_returns(self):
        for banker in range(4):
            for score in (0, 160):
                with self.subTest(banker=banker, score=score):
                    env, meta = self.play_trick([[9], [13], [17], [21]], banker, score, 'A')
                    game = env._env
                    self.assertEqual(game.getStage(), 'finalend')
                    self.assertEqual(game.getLastBanker(), banker)
                    self.assertEqual(meta['winner_role'], game._role_of(3, banker))
                    rewards = env._get_reward()
                    for seat in range(4):
                        role = game._role_of(seat, banker)
                        self.assertEqual(rewards[role], game.getEndingScore(seat))
                        self.assertEqual(self.label(meta, role) == 5, seat % 2 == 3 % 2)

    def test_nonterminal_roles_use_current_banker(self):
        env, meta = self.play_trick([[9, 20], [13, 12], [17, 22], [21, 23]], banker=1)
        self.assertEqual(env._env.getStage(), 'roundend')
        self.assertEqual(meta['winner_role'], 'banker_op')
        self.assertEqual(meta['roles']['banker_up']['is_lead'], 1.)
        self.assertEqual(meta['is_last_round'], 0.)

    def test_second_deck_same_suit_is_following(self):
        _, meta = self.play_trick([[9, 20], [63, 12], [13, 22], [17, 23]])
        self.assertEqual(meta['roles']['banker_down']['followed'], 1.)
        self.assertEqual(self.label(meta, 'banker_down'), 4)

    def test_second_deck_different_suit_is_not_following(self):
        _, meta = self.play_trick([[9, 20], [65, 12], [13, 22], [17, 23]])
        self.assertEqual(meta['roles']['banker_down']['followed'], 0.)
        self.assertEqual(self.label(meta, 'banker_down'), 4)

    def test_successful_ruff_is_not_discard(self):
        for last_card, expected_label in ((17, 1), (21, 0)):
            with self.subTest(last_card=last_card):
                _, meta = self.play_trick([[9, 20], [8, 12], [13, 22], [last_card, 23]])
                self.assertEqual(meta['winner_role'], 'banker_down')
                self.assertEqual(meta['roles']['banker_down']['followed'], 0.)
                self.assertEqual(self.label(meta, 'banker_down'), expected_label)

    def test_terminal_teammate_win_takes_priority_over_discard(self):
        teams = [('banker', 'banker_op'), ('banker_down', 'banker_up')]
        for team in teams:
            for winner, teammate in (team, team[::-1]):
                for followed in (0., 1.):
                    with self.subTest(winner=winner, teammate=teammate, followed=followed):
                        meta = {
                            'is_last_round': 1., 'winner_role': winner,
                            'roles': {teammate: {
                                'is_lead': 0., 'threw': 0., 'followed': followed, 'won': 0.,
                            }},
                        }
                        self.assertEqual(self.label(meta, teammate), 5)
                        meta['is_last_round'] = 0.
                        self.assertEqual(self.label(meta, teammate), 4)

    def test_discard_includes_following_and_not_following_when_opponent_wins(self):
        for second_card in (13, 65):
            with self.subTest(second_card=second_card):
                # Seat 2 wins for the opposing team; seat 1 loses with either h4 or d3.
                _, meta = self.play_trick([[9, 20], [second_card, 12], [21, 22], [17, 23]])
                self.assertEqual(meta['winner_role'], 'banker_op')
                self.assertEqual(meta['roles']['banker_down']['followed'], float(second_card == 13))
                self.assertEqual(self.label(meta, 'banker_down'), 4)

    def test_winning_follower_and_losing_leader_are_not_discard(self):
        for last_card, expected_label in ((17, 1), (21, 0)):
            with self.subTest(last_card=last_card):
                _, meta = self.play_trick([[9, 20], [25, 12], [13, 22], [last_card, 23]])
                self.assertEqual(meta['winner_role'], 'banker_down')
                self.assertEqual(meta['roles']['banker_down']['followed'], 1.)
                self.assertEqual(self.label(meta, 'banker_down'), expected_label)
                self.assertEqual(self.label(meta, 'banker'), 3)

    def test_nonterminal_leader_can_still_be_teammate_assist(self):
        _, meta = self.play_trick([[9, 20], [13, 12], [25, 22], [17, 23]])
        self.assertEqual(meta['winner_role'], 'banker_op')
        self.assertEqual(self.label(meta, 'banker'), 2)


if __name__ == '__main__':
    unittest.main()
