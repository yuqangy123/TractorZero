import copy
import gc
import importlib
import queue
import random
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rlcard.games.tractors.act import get_batch_play
from rlcard.games.tractors.dmc.dmc import learn_play
from rlcard.games.tractors.env.env_utils import Environment
from rlcard.games.tractors.env.utils import __PLAY_ROLES__, __SINGLE__, __WRONG__
from rlcard.games.tractors.model.models import Model
from rlcard.games.tractors.model.play_model import BankerModel, IdlerModel, NUM_STRATEGY


def make_batch():
    # T=2, B=2; chosen options intentionally differ from both argmax Q and labels.
    return {
        'obs_strategy_z': torch.zeros(2, 2, 297),
        'obs_strategy_x': torch.zeros(2, 2, 58, 4, 15),
        'obs_action_type_z': torch.zeros(2, 2, 303),
        'obs_action_type_x': torch.zeros(2, 2, 58, 4, 15),
        'obs_z': torch.zeros(2, 2, 303),
        'obs_x': torch.zeros(2, 2, 60, 4, 15),
        'strategy_action': torch.tensor([[1, 6], [1, 6]]),
        'strategy_target': F.one_hot(torch.full((2, 2), 3), NUM_STRATEGY).float(),
        'target_wp': torch.tensor([[1., -1.], [1/3, -1/3]]),
        'target_adp': torch.tensor([[0.25, -0.25], [0.125, -0.125]]),
    }


class SmallPlayModel(nn.Module):
    """Expose the selected Q gradient and exactly which condition each lower head sees."""
    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.tensor([0.9, 0., 0., 0., 0., 0., 0.]))
        self.lower = nn.Parameter(torch.zeros(3))
        self.conditions = []

    def forward_strategy(self, z, x, return_value=False, flags=None):
        return {'values': self.q.expand(z.shape[0], -1)}

    def forward_tp(self, z, x, strategy, return_value=False, flags=None):
        self.conditions.append(strategy.detach().clone())
        return {'values': self.lower.expand(z.shape[0], -1).split(1, dim=-1)}

    forward_act = forward_tp


class TestStrategyLearning(unittest.TestCase):
    flags = SimpleNamespace(training_device='cpu', max_grad_norm=20.)

    def train_once(self, model, batch, actor_models=None):
        return learn_play('banker', actor_models or {}, model, batch,
                          torch.optim.SGD(model.parameters(), lr=0.1), None,
                          self.flags, threading.Lock())

    def test_return_updates_only_executed_strategies_and_preserves_conditions(self):
        model = SmallPlayModel()
        actor_copy = copy.deepcopy(model)
        actors = {'cpu': SimpleNamespace(get_model=lambda position: actor_copy)}
        batch = make_batch()
        before = model.q.detach().clone()
        stats = self.train_once(model, batch, actors)
        expected_condition = F.one_hot(batch['strategy_action'].flatten(), NUM_STRATEGY).float()
        for condition in model.conditions:
            torch.testing.assert_close(condition, expected_condition)
            self.assertFalse(condition.requires_grad)
        self.assertGreater(model.q[1].item(), before[1].item())
        self.assertLess(model.q[6].item(), before[6].item())
        untouched = [0, 2, 3, 4, 5]
        torch.testing.assert_close(model.q[untouched], before[untouched])
        self.assertEqual(model.q.grad[untouched].abs().sum().item(), 0.)
        self.assertAlmostEqual(stats['loss_strategy_banker'], 5/9, places=6)
        self.assertEqual(stats['strategy_sel_banker_1'], 0.5)
        self.assertEqual(stats['strategy_sel_banker_6'], 0.5)
        torch.testing.assert_close(actor_copy.q, model.q)

    def test_diagnostic_labels_do_not_change_any_training_gradient(self):
        first, second = SmallPlayModel(), SmallPlayModel()
        batch = make_batch()
        self.train_once(first, batch)
        batch['strategy_target'] = F.one_hot(torch.zeros(2, 2, dtype=torch.long), NUM_STRATEGY).float()
        self.train_once(second, batch)
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right)
            torch.testing.assert_close(left.grad, right.grad)

    def test_single_sample_win_only_or_loss_only_is_finite(self):
        for reward in (1., -1.):
            batch = {k: v[:1, :1] for k, v in make_batch().items()}
            batch['target_wp'] = torch.full((1, 1), reward)
            stats = self.train_once(SmallPlayModel(), batch)
            self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_queue_batch_preserves_time_batch_strategy_alignment(self):
        chunks = queue.Queue()
        original = make_batch()
        for column in range(2):
            chunks.put({key: value[:, column] for key, value in original.items()})
        batch = get_batch_play(chunks, SimpleNamespace(batch_size=2), threading.Lock())
        for key, value in original.items():
            torch.testing.assert_close(batch[key], value)

    def test_real_banker_and_idler_forward_and_backward(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            for model_type in (BankerModel, IdlerModel):
                with self.subTest(model_type=model_type.__name__):
                    model = model_type(__WRONG__, torch.device('cpu'))
                    model.eval()
                    z, x = torch.randn(1, 297), torch.randn(1, 58, 4, 15)
                    with torch.no_grad():
                        out = model.forward_strategy(z, x)
                        self.assertEqual(out['action'].ndim, 0)
                        self.assertEqual(out['action'].item(), out['values'][0].argmax().item())
                        with patch('numpy.random.rand', return_value=0.), patch(
                                'torch.randint', return_value=torch.tensor(4)):
                            explored = model.forward_strategy(z, x, flags=SimpleNamespace(exp_epsilon=1.))
                        self.assertEqual(explored['action'].item(), 4)
                        with self.assertRaisesRegex(ValueError, 'exactly one state'):
                            model.forward_strategy(z.expand(2, -1), x.expand(2, -1, -1, -1))
                    batch = {k: v[:, :1] for k, v in make_batch().items()}
                    # Both returns attach to option 1; use nonzero observations for encoder gradients.
                    for key in ('obs_strategy_z', 'obs_strategy_x', 'obs_action_type_z',
                                'obs_action_type_x', 'obs_z', 'obs_x'):
                        batch[key] = torch.randn_like(batch[key])
                    model.train()
                    last_layer = model.strategy_net[-1]
                    before = last_layer.weight[1].detach().clone()
                    stats = self.train_once(model, batch)
                    self.assertTrue(all(np.isfinite(value) for value in stats.values()))
                    self.assertGreater(last_layer.weight.grad[1].abs().sum().item(), 0.)
                    self.assertFalse(torch.equal(before, last_layer.weight[1]))
                    unselected = [0, 2, 3, 4, 5, 6]
                    self.assertEqual(last_layer.weight.grad[unselected].abs().sum().item(), 0.)
                    del model, last_layer, before
                    gc.collect()
        finally:
            torch.set_num_threads(old_threads)


class TestStrategyInference(unittest.TestCase):
    def test_plots_do_not_treat_strategy_ids_as_expert_label_names(self):
        from rlcard.games.tractors.eval import plot_results

        data = {'frames': np.array([1, 2]),
                'sel': np.full((2, NUM_STRATEGY), 1 / NUM_STRATEGY),
                'tgt': np.full((2, NUM_STRATEGY), 1 / NUM_STRATEGY)}
        with patch.object(plot_results.os.path, 'exists', return_value=True), \
                patch.object(plot_results.np, 'load', return_value=data), \
                patch.object(plot_results.plt.Figure, 'tight_layout'), \
                patch.object(plot_results.plt.Figure, 'savefig', autospec=True) as save:
            plot_results.plot_strategy_distribution('unused', 'unused')
        self.assertEqual(save.call_count, 2 * len(__PLAY_ROLES__))
        distribution, series = (save.call_args_list[i].args[0] for i in range(2))
        self.assertEqual(len(distribution.axes), 2)
        option_names = [f'g{i + 1}' for i in range(NUM_STRATEGY)]
        self.assertEqual([t.get_text() for t in distribution.axes[0].get_xticklabels()], option_names)
        self.assertIn('diagnostic', distribution.axes[1].get_title())
        self.assertEqual([line.get_label() for line in series.axes[0].lines], option_names)

    def test_one_chosen_strategy_conditions_type_and_all_action_chunks(self):
        class RecordingModel:
            def __init__(self):
                self.strategy_calls = 0
                self.conditions = []

            def forward_strategy(self, z, x, flags=None):
                self.strategy_calls += 1
                self.asserted_shapes = (tuple(z.shape), tuple(x.shape))
                return {'action': torch.tensor(5)}

            def forward_tp(self, z, x, strategy, flags=None):
                self.conditions.append(strategy.clone())
                return {'action': torch.tensor(0)}

            def forward_act(self, z, x, strategy, **kwargs):
                self.conditions.append(strategy.clone())
                value = torch.arange(x.shape[0], dtype=torch.float32).unsqueeze(-1)
                return {'values': (torch.ones_like(value), value, value)}

        recorder = RecordingModel()
        wrapper = Model.__new__(Model)
        wrapper._device = torch.device('cpu')
        wrapper.models = {'banker': recorder}
        actions = [np.zeros((300, 2, 4, 15), dtype=np.float32)]
        output = wrapper.play('banker', torch.zeros(297), torch.zeros(58, 4, 15),
                              torch.zeros(2, 303), torch.zeros(2, 58, 4, 15), actions, [0, 1])
        self.assertEqual(recorder.strategy_calls, 1)
        self.assertEqual(recorder.asserted_shapes, ((1, 297), (1, 58, 4, 15)))
        self.assertEqual([c.shape[0] for c in recorder.conditions], [2, 256, 44])
        expected = F.one_hot(torch.tensor(5), NUM_STRATEGY).float()
        torch.testing.assert_close(output['strategy'], expected)
        for condition in recorder.conditions:
            torch.testing.assert_close(condition, expected.expand(condition.shape[0], -1))

    def test_actor_keeps_behavior_strategy_and_game_return_across_unrolls(self):
        act_module = importlib.import_module('rlcard.games.tractors.act')
        selected = {p: [] for p in __PLAY_ROLES__}
        expected_returns = {p: [] for p in __PLAY_ROLES__}
        chunks = {p: [] for p in __PLAY_ROLES__ + ['bid', 'cover']}

        class StopActor(BaseException):
            pass

        class RecordingEnvironment(Environment):
            def step(self, action):
                self.test_steps = getattr(self, 'test_steps', 0) + 1
                if self.test_steps > 2000:
                    raise AssertionError('Actor did not produce two unrolls in 2000 environment steps')
                position, obs, output = super().step(action)
                if 'game_reward' in output:
                    for role in __PLAY_ROLES__:
                        missing = len(selected[role]) - len(expected_returns[role])
                        expected_returns[role].extend([output['game_reward'][role] / 3.] * missing)
                return position, obs, output

        class SamplingModel:
            def bid(self, z, x, flags=None):
                return {'action': torch.tensor(0)}

            def cover(self, z, x, flags=None):
                return {'action': torch.ones(1, 120)}

            def play(self, role, zs, xs, z, x, legal_actions, legal_types, flags=None):
                strategy = (len(selected[role]) + __PLAY_ROLES__.index(role)) % NUM_STRATEGY
                selected[role].append(strategy)
                type_index = legal_types.index(__SINGLE__) if __SINGLE__ in legal_types else 0
                return {'action_type_index': type_index, 'action': 0,
                        'strategy': F.one_hot(torch.tensor(strategy), NUM_STRATEGY).float()}

        class RecordingQueue:
            def __init__(self, role):
                self.role = role

            def put(self, chunk):
                chunks[self.role].append(chunk)
                if all(len(chunks[role]) >= 2 for role in __PLAY_ROLES__):
                    raise StopActor()

        flags = SimpleNamespace(unroll_length=17, training_device='cpu', exp_epsilon=0.25, objective='adp')
        counter = lambda: SimpleNamespace(value=0, get_lock=threading.Lock)
        random_state, numpy_state = random.getstate(), np.random.get_state()
        random.seed(0)
        np.random.seed(0)
        try:
            with patch.object(act_module, 'Environment', RecordingEnvironment):
                with self.assertRaises(StopActor):
                    act_module.act(0, 'cpu', {p: RecordingQueue(p) for p in chunks},
                                   SamplingModel(), counter(), counter(), SimpleNamespace(value=0.25), flags)
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)
        for role in __PLAY_ROLES__:
            self.assertEqual(len(chunks[role]), 2)
            recorded_actions = torch.cat([c['strategy_action'] for c in chunks[role]])
            recorded_returns = torch.cat([c['target_wp'] for c in chunks[role]])
            self.assertEqual(recorded_actions.dtype, torch.long)
            self.assertEqual(recorded_actions.tolist(), selected[role][:34])
            torch.testing.assert_close(recorded_returns, torch.tensor(expected_returns[role][:34]))
            for chunk in chunks[role]:
                self.assertTrue(all(v.shape[0] == 17 for v in chunk.values()))


if __name__ == '__main__':
    unittest.main()
