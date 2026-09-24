import unittest

from rlcard.games.tractors.env.botzone import tractorGame
from rlcard.games.tractors.env.env import Env
from rlcard.games.tractors.env.env_utils import Environment


def deal(offset=0):
    cards = list(range(offset, 108)) + list(range(offset))
    return {'allocation': [cards[i * 25:(i + 1) * 25] for i in range(4)],
            'publiccard': cards[100:]}


class TestTractorsSeriesReset(unittest.TestCase):
    def test_finalend_starts_new_series_and_preserves_deal_source(self):
        for banker in range(4):
            for score in (0, 160):
                with self.subTest(banker=banker, score=score):
                    env = Env()
                    game = env._env
                    deals = {'2': [deal(), deal(4)], 'A': [deal(8)]}
                    game.set_eval_deals(deals)
                    game._game_log = False
                    env.reset()
                    game.player_level = ['A'] * 4
                    game.globalInfo['level'] = 'A'
                    game.globalInfo['banking']['banker'] = banker
                    game.setMajor('s', 'A')
                    game.get_score = 10
                    game.get_score_pok = [36]
                    game.EndGame(banker, score)
                    self.assertEqual(game.getStage(), 'finalend')

                    obs = env.reset()

                    self.assertEqual(obs['position'], 'bid')
                    self.assertEqual(env.infoset.level, '2')
                    self.assertEqual(game.player_level, ['2'] * 4)
                    self.assertEqual(game.total_score, [0] * 4)
                    self.assertTrue(game.globalInfo['first_round'])
                    self.assertEqual(game.getBanker(), -1)
                    self.assertEqual(game.getPlayerPosition(), 0)
                    self.assertNotIn('ending_score', game.globalInfo)
                    self.assertEqual(game.getLastRoundScore(), 0)
                    self.assertEqual(game.get_score_pok, [])
                    self.assertEqual(game.getPlayHistory(), [])
                    self.assertEqual(game.player_played_cards, [[], [], [], []])
                    self.assertEqual(game.globalInfo['deal_count'], [1, 0, 0, 0])
                    self.assertIs(game._eval_deals, deals)
                    self.assertEqual(game._eval_deal_idx, {'2': 2, 'A': 0})
                    self.assertEqual(game.getPlayerHandCards(0), [4])
                    self.assertIs(game._game_log, False)
                    del game._game_log
                    # New-series bidding must be able to establish the banker again.
                    env.step(['s'])
                    self.assertEqual(game.getBanker(), 0)

    def test_gameend_keeps_levels_scores_and_next_banker(self):
        for score in (0, 80, 120):
            with self.subTest(score=score):
                env = Env()
                game = env._env
                game.set_eval_deals({level: [deal()] * 2 for level in ('2', '5', '6', '8')})
                env.reset()
                game.player_level = ['5'] * 4
                game.globalInfo['level'] = '5'
                game.globalInfo['banking']['banker'] = 1
                game.EndGame(1, score)
                game.globalInfo['stage'] = 'gameend'
                levels = game.player_level[:]
                scores = game.total_score[:]
                level, banker = game.getLevel(), game.getBanker()

                env.reset()

                self.assertEqual(game.player_level, levels)
                self.assertEqual(game.total_score, scores)
                self.assertEqual(env.infoset.level, level)
                self.assertEqual(game.getBanker(), banker)
                self.assertEqual(game.getPlayerPosition(), banker)
                self.assertFalse(game.globalInfo['first_round'])

    def test_training_wrapper_preserves_terminal_reward_before_reset(self):
        env = Env()
        wrapper = Environment(env, 'cpu')
        wrapper.initial(None, 'cpu')
        game = env._env
        game.setMajor('s', 'A')
        game.player_level = ['A'] * 4
        game.globalInfo = {
            'stage': 'play', 'level': 'A', 'game_score': 160,
            'banking': {'banker': 0, 'banker_last': 0, 'major': 's'},
            'publiccard': [], 'history': [[], [], 0, 0], 'playerpos': 0,
        }
        game.player_hand_cards = [[9], [13], [17], [21]]
        for seat in range(3):
            tractorGame.step(game, [seat, [game.player_hand_cards[seat][0]]])

        position, obs, output = wrapper.step([[21], 1])

        self.assertTrue(output['done'])
        self.assertEqual(output['game_score'], 165)
        self.assertEqual(output['game_reward']['banker'], -3)
        self.assertEqual(output['game_reward']['banker_up'], 3)
        self.assertEqual(output['step_meta']['winner_role'], 'banker_up')
        self.assertEqual(output['step_meta']['is_last_round'], 1.)
        self.assertEqual((position, output['stage']), ('bid', 'bid'))
        self.assertEqual(env.infoset.level, '2')
        self.assertEqual(game.player_level, ['2'] * 4)
        self.assertEqual(tuple(obs['z'].shape), (20,))


if __name__ == '__main__':
    unittest.main()
