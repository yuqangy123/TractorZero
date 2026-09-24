import copy
import importlib
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from rlcard.games.tractors.dmc.dmc import learn_cover, terminal_value_loss
from rlcard.games.tractors.env.utils import __PLAY_ROLES__, cards2matrix, matrix2cards
from rlcard.games.tractors.model.bid_model import BidModel
from rlcard.games.tractors.model.cover_model import CoverModel, select_cover_action
from rlcard.games.tractors.model.checkpoint import (
    TRAINING_SCHEMA_VERSION, load_warm_start, require_current_checkpoint,
)
from tests.models.test_tractors_strategy import SmallPlayModel, make_batch, TestStrategyLearning


class TestDecisionTargets(unittest.TestCase):
    def test_bid_greedy_selects_best_candidate_not_singleton_axis(self):
        class Features(nn.Module):
            def forward(self, x):
                return x
        model = BidModel.__new__(BidModel)
        nn.Module.__init__(model)
        model.resnet = Features()
        for name in ('dense1', 'dense2', 'dense3', 'dense4', 'dense5', 'dense_action'):
            setattr(model, name, nn.Identity())
        for values, expected in (([.1, .9, .2], 1), ([.9, .1], 0), ([.1], 0)):
            out = model(torch.empty(len(values), 0), torch.tensor(values).unsqueeze(-1))
            self.assertEqual(out['action'].item(), expected)

    def test_binary_probability_and_conditional_terminal_values_have_zero_loss(self):
        returns = torch.tensor([1/3, -2/3, 1., -1.])
        values = tuple(v.unsqueeze(-1) for v in (returns.sign(), returns, returns))
        self.assertEqual(terminal_value_loss(values, returns).item(), 0.)
        probability = (values[0].squeeze(-1) + 1) / 2
        torch.testing.assert_close(probability, torch.tensor([1., 0., 1., 0.]))
        reconstructed = probability * values[1].squeeze(-1) + (1 - probability) * values[2].squeeze(-1)
        torch.testing.assert_close(reconstructed, returns)

    def test_trick_scores_do_not_change_training_gradient(self):
        learner = TestStrategyLearning()
        first, second = SmallPlayModel(), SmallPlayModel()
        batch = make_batch()
        learner.train_once(first, batch)
        batch['target_adp'] = torch.full_like(batch['target_adp'], 100.)
        learner.train_once(second, batch)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a.grad, b.grad)

    def test_cover_mask_excludes_unheld_and_padding_cards_for_negative_values(self):
        hand = [0, 4, 8, 12, 16, 20, 24, 28, 52, 53, 106, 107]
        legal = torch.from_numpy(cards2matrix(hand))
        scores = torch.zeros(120)
        scores[legal.flatten().bool()] = -torch.arange(1, len(hand) + 1).float()
        action = select_cover_action(scores, legal)
        selected = matrix2cards(action).tolist()
        self.assertEqual(len(selected), 8)
        self.assertTrue(set(selected).issubset(hand))
        with self.assertRaises(ValueError):
            select_cover_action(scores, torch.zeros_like(legal))

    def test_real_cover_values_can_fit_negative_returns(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = CoverModel().eval()
            with torch.no_grad():
                model.dense4.weight.zero_()
                model.dense4.bias.fill_(-2.)
                output = model(torch.zeros(1, 16), torch.zeros(1, 10, 4, 15), return_value=True)
            self.assertTrue((output['values'][0] < -.9).all())
            torch.testing.assert_close(output['values'][0], output['action'])
        finally:
            torch.set_num_threads(old_threads)

    def test_cover_learning_updates_selected_cards_with_signed_return(self):
        class SmallCover(nn.Module):
            def __init__(self):
                super().__init__()
                self.values = nn.Parameter(torch.zeros(120))
            def forward(self, z, x, return_value=False):
                return {'values': (self.values.tanh().unsqueeze(0).expand(z.shape[0], -1),)}
        for reward in (-1., 1.):
            model = SmallCover()
            mask = torch.zeros(1, 1, 120)
            mask[:, :, :8] = 1
            batch = {'obs_z': torch.zeros(1, 1, 16), 'obs_x': torch.zeros(1, 1, 10, 4, 15),
                     'cover_action_mask': mask, 'target_wp': torch.tensor([[reward]])}
            stats = learn_cover('cover', {}, model, batch, torch.optim.SGD(model.parameters(), lr=.1),
                                None, SimpleNamespace(training_device='cpu', max_grad_norm=20.), threading.Lock())
            self.assertEqual(stats['loss_cover'], 1.)
            self.assertTrue((model.values[:8] * reward > 0).all())
            self.assertEqual(model.values[8:].abs().sum().item(), 0.)


class TestActorEpisodeCredit(unittest.TestCase):
    def test_bid_seats_and_cover_returns_align_across_multiple_unrolls(self):
        module = importlib.import_module('rlcard.games.tractors.act')
        chunks = {p: [] for p in __PLAY_ROLES__ + ['bid', 'cover']}
        expected = {'bid': [], 'cover': []}
        decisions = []
        class StopActor(BaseException):
            pass
        class ScriptedEnvironment:
            def __init__(self, env, device):
                self.episode, self.index = 1, 0
                self.env = SimpleNamespace(_env=SimpleNamespace(getPlayerPosition=lambda: [1, 2, 3, 0][self.index]))
            def observation(self):
                cover = self.index == 3
                n = 1 if cover or self.index == 2 else 2
                marker = self.episode * 10 + self.index
                obs = {'z': torch.full((16 if cover else 20,), marker),
                       'x': torch.full((10 if cover else 8, 4, 15), marker),
                       'z_batch': torch.full((n, 16 if cover else 20), marker),
                       'x_batch': torch.full((n, 10, 4, 15), marker)}
                legal = torch.from_numpy(cards2matrix(list(range(33)))) if cover else [['', []], ['s', []]][:n]
                stage = 'cover' if cover else 'bid'
                return stage, obs, {'stage': stage, 'legal_actions': legal}
            def initial(self, *args, **kwargs):
                return self.observation()
            def step(self, action):
                marker = self.episode * 10 + self.index
                if self.index < 2:
                    decisions.append((marker, [1, 2][self.index], action[0]))
                if self.index < 3:
                    self.index += 1
                    return self.observation()
                banker = self.episode % 4
                grade = 3 if self.episode % 2 else -1
                for bid_marker, seat, _ in decisions[-2:]:
                    expected['bid'].append((bid_marker, grade / 3 * (1 if (seat - banker) % 2 == 0 else -1)))
                expected['cover'].append((marker, float(np.sign(grade))))
                self.episode += 1
                self.index = 0
                pos, obs, extra = self.observation()
                extra.update(game_reward={'banker': grade, 'banker_op': grade,
                                          'banker_down': -grade, 'banker_up': -grade,
                                          'cover': float(np.sign(grade))}, banker_seat=banker)
                return pos, obs, extra
        class Model:
            def bid(self, z, x, flags=None):
                # Alternate between actual bidding and voluntarily passing.
                return {'action': torch.tensor(1 if len(z) == 2 and int(z[0, 0]) % 10 == 0 else 0)}
            def cover(self, *args, **kwargs):
                return {'action': -torch.arange(1, 121).float().unsqueeze(0)}
        class RecordingQueue:
            def __init__(self, role):
                self.role = role
            def put(self, chunk):
                chunks[self.role].append(chunk)
                if len(chunks['cover']) >= 2 and len(chunks['bid']) >= 3:
                    raise StopActor()
        flags = SimpleNamespace(unroll_length=3, objective='adp', exp_epsilon=0., training_device='cpu')
        counter = lambda: SimpleNamespace(value=0, get_lock=threading.Lock)
        with patch.object(module, 'Environment', ScriptedEnvironment):
            with self.assertRaises(StopActor):
                module.act(0, 'cpu', {p: RecordingQueue(p) for p in chunks}, Model(),
                           counter(), counter(), SimpleNamespace(value=0.), flags)
        for role in ('bid', 'cover'):
            values = torch.cat([b['target_wp'] for b in chunks[role]])
            markers = torch.cat([b['obs_z'] for b in chunks[role]])[:, 0]
            wanted = expected[role][:len(values)]
            self.assertEqual(markers.tolist(), [m for m, _ in wanted])
            torch.testing.assert_close(values, torch.tensor([v for _, v in wanted]))
        self.assertEqual({a for _, _, a in decisions}, {'', 's'})


class TestEvaluationAndCheckpoints(unittest.TestCase):
    def test_eval_mode_survives_loading_and_old_checkpoints_are_rejected(self):
        from rlcard.games.tractors.agents import hrl_agent
        class TinyModel:
            def __init__(self, **kwargs):
                self.network = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 1))
            def eval(self):
                self.network.eval()
            def get_model(self, position):
                return self.network
        with tempfile.TemporaryDirectory() as tmp, patch.object(hrl_agent, 'Model', TinyModel):
            agent = hrl_agent.HRLAgent('cpu')
            self.assertFalse(agent._model.network.training)
            path = Path(tmp) / 'banker.ckpt'
            torch.save({'training_schema_version': TRAINING_SCHEMA_VERSION,
                        'model_state_dict': agent._model.network.state_dict()}, path)
            agent._model.network.train()
            agent.loadModel('banker', path, 'cpu')
            self.assertFalse(agent._model.network.training)
            # Running BN statistics must remain frozen during evaluation.
            before = agent._model.network[0].running_mean.clone()
            with torch.no_grad():
                agent._model.network(torch.randn(3, 2))
            torch.testing.assert_close(before, agent._model.network[0].running_mean)
            torch.save(agent._model.network.state_dict(), path)
            with self.assertRaisesRegex(ValueError, 'warm_start_from'):
                agent.loadModel('banker', path, 'cpu')
            with self.assertRaises(FileNotFoundError):
                agent.loadModel('banker', Path(tmp) / 'missing', 'cpu')

    def test_warm_start_migrates_features_and_keeps_new_heads(self):
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.type_q_net = nn.Sequential(nn.Linear(5026, 2, bias=False))
                # Match the nested residual block parameter name without allocating the real model.
                self.type_q_net[0] = nn.Module()
                self.type_q_net[0].fc = nn.Linear(5026, 2, bias=False)
                self.strategy_net = nn.Linear(2, 7)
        model = SmallModel()
        before = copy.deepcopy(model.state_dict())
        old = copy.deepcopy(before)
        old['type_q_net.0.fc.weight'] = torch.ones(2, 5005)
        old['strategy_net.weight'].zero_()
        load_warm_start(model, old, legacy=True)
        self.assertTrue((model.type_q_net[0].fc.weight[:, :5005] == 1).all())
        torch.testing.assert_close(model.type_q_net[0].fc.weight[:, 5005:], before['type_q_net.0.fc.weight'][:, 5005:])
        torch.testing.assert_close(model.strategy_net.weight, before['strategy_net.weight'])
        with self.assertRaisesRegex(ValueError, 'old learning targets'):
            require_current_checkpoint({})
        require_current_checkpoint({'training_schema_version': TRAINING_SCHEMA_VERSION})


if __name__ == '__main__':
    unittest.main()
