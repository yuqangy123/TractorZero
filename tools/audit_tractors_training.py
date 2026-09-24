"""Read-only training audit; does not load checkpoints or start training.

Run from the repository root with the rlcard Python environment:
    python tools/audit_tractors_training.py
Outputs a JSON evidence file under docs/audits.
"""
import copy
import csv
import gc
import importlib
import json
import random
import sys
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from rlcard.games.tractors.env.env import Env
from rlcard.games.tractors.env.env_utils import Environment
from rlcard.games.tractors.env.utils import __PLAY_ROLES__, __SINGLE__, cards2matrix, matrix2cards


def log_evidence():
    rows = list(csv.DictReader((ROOT / 'log/eval_results/history.csv').open()))
    result = {'winrates': {}, 'losses': {}, 'strategy_snapshots': {}}
    for tag in sorted({r['tag'] for r in rows}):
        rr = sorted([r for r in rows if r['tag'] == tag], key=lambda r: int(r['step']))
        key = 'banker_winrate' if tag.startswith('banker') else 'idler_winrate'
        def aggregate(rs):
            n = sum(int(r['num_games']) for r in rs)
            return {'games': n, 'rate': sum(float(r[key]) * int(r['num_games']) for r in rs) / n}
        result['winrates'][tag] = {
            'checkpoints': len(rr), 'first_step': int(rr[0]['step']), 'last_step': int(rr[-1]['step']),
            'first5': aggregate(rr[:5]), 'last5': aggregate(rr[-5:]),
            'first_rate': float(rr[0][key]), 'last_rate': float(rr[-1][key]),
            'last_games': int(rr[-1]['num_games']),
        }
    for role in __PLAY_ROLES__:
        acc = EventAccumulator(str(ROOT / 'log/tractors_loss' / role), size_guidance={'scalars': 0})
        acc.Reload()
        result['losses'][role] = {}
        for tag in acc.Tags()['scalars']:
            if not tag.startswith('loss'):
                continue
            ss = acc.Scalars(tag)
            result['losses'][role][tag] = {
                'count': len(ss), 'first_step': ss[0].step, 'last_step': ss[-1].step,
                'first': ss[0].value, 'last': ss[-1].value,
                'maximum': max(s.value for s in ss),
                'nonfinite_count': sum(not np.isfinite(s.value) for s in ss),
            }
        with np.load(ROOT / 'log/strategy_dist' / (role + '.npz')) as data:
            result['strategy_snapshots'][role] = {
                'rows': len(data['frames']), 'first_frame': int(data['frames'][0]),
                'last_frame': int(data['frames'][-1]),
                'selection_mean': data['sel'].mean(axis=0).tolist(),
                'label_mean': data['tgt'].mean(axis=0).tolist(),
            }
    return result


def model_evidence():
    from rlcard.games.tractors.model.bid_model import BidModel
    from rlcard.games.tractors.model.cover_model import CoverModel
    from rlcard.games.tractors.model.play_model import BankerModel, NUM_STRATEGY
    from rlcard.games.tractors.model.BasicBlockM import ResNet
    result = {}
    # Exercise the real BidModel selection code with deliberately ranked candidates.
    class FixedFeatures(nn.Module):
        def forward(self, x):
            return torch.tensor([[0.1], [0.9], [0.2]])
    bid = BidModel.__new__(BidModel)
    nn.Module.__init__(bid)
    bid.resnet = FixedFeatures()
    for name in ['dense1', 'dense2', 'dense3', 'dense4', 'dense5', 'dense_action']:
        setattr(bid, name, nn.Identity())
    out = bid(torch.empty(3, 0), torch.empty(3, 1), return_value=True)
    result['bid_argmax'] = {'scores': out['values'][0].flatten().tolist(),
                            'selected': out['action'].item(), 'correct': 1}
    # Real play network, random weights: isolate dependency on strategy and BN mode.
    torch.manual_seed(20260920)
    model = BankerModel(6, torch.device('cpu'))
    result['play_parameter_count'] = sum(p.numel() for p in model.parameters())
    result['encoder_parameter_count'] = sum(p.numel() for p in model.cards_encoder_action.parameters())
    z, x = torch.randn(3, 303), torch.randn(3, 60, 4, 15)
    goals = F.one_hot(torch.tensor([0, 0, 0]), NUM_STRATEGY).float()
    goals2 = F.one_hot(torch.tensor([6, 6, 6]), NUM_STRATEGY).float()
    model.eval()
    with torch.no_grad():
        a = torch.cat(model.forward_act(z, x, goals, return_value=True)['values'], -1)
        b = torch.cat(model.forward_act(z, x, goals2, return_value=True)['values'], -1)
        result['strategy_output_max_difference'] = (a - b).abs().max().item()
        a = torch.cat(model.forward_tp(z, x[:, :58], goals, return_value=True)['values'], -1)
        b = torch.cat(model.forward_tp(z, x[:, :58], goals2, return_value=True)['values'], -1)
        result['strategy_type_max_difference'] = (a - b).abs().max().item()
        state = copy.deepcopy(model.state_dict())
        model.train()
        first = torch.cat(model.forward_act(z[:1], x[:1], goals[:1], return_value=True)['values'], -1)
        model.load_state_dict(state)
        batch = torch.cat(model.forward_act(z, x, goals, return_value=True)['values'], -1)[:1]
        result['train_mode_candidate_batch_max_difference'] = (first - batch).abs().max().item()
        model.load_state_dict(state)
        model.eval()
        first = torch.cat(model.forward_act(z[:1], x[:1], goals[:1], return_value=True)['values'], -1)
        batch = torch.cat(model.forward_act(z, x, goals, return_value=True)['values'], -1)[:1]
        result['eval_mode_candidate_batch_max_difference'] = (first - batch).abs().max().item()
    del model, state
    gc.collect()
    # Confirm HRLAgent keeps the default train mode, without allocating six full models.
    from rlcard.games.tractors.agents import hrl_agent
    class TinyModel:
        def __init__(self, **kwargs):
            self.net = nn.BatchNorm1d(2)
        def eval(self):
            self.net.eval()
    with patch.object(hrl_agent, 'Model', TinyModel):
        agent = hrl_agent.HRLAgent('cpu')
        result['hrl_agent_default_training'] = agent._model.net.training
    # Exercise actual cover forward/loss with a minimal trainable head.
    class ZeroFeatures(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1)
    cover = CoverModel.__new__(CoverModel)
    nn.Module.__init__(cover)
    cover.resnet = ZeroFeatures()
    cover.dense1 = cover.dense2 = cover.dense3 = nn.Identity()
    cover.dense4 = nn.Linear(1, 120)
    nn.init.zeros_(cover.dense4.weight)
    nn.init.zeros_(cover.dense4.bias)
    output = cover(torch.empty(1, 0), torch.empty(1, 1), return_value=True)['values'][0]
    loss = (output[:, :8] + 1).square().sum() / 120
    loss.backward()
    result['cover_negative_reward'] = {'output_at_zero_logits': output[0, 0].item(),
        'loss_at_zero_logits': loss.item(), 'infimum_for_8_selected_cards': 8 / 120,
        'selected_logit_gradient': cover.dense4.bias.grad[0].item()}
    return result


def actor_alignment_evidence():
    act_module = importlib.import_module('rlcard.games.tractors.act')
    roles = __PLAY_ROLES__ + ['bid', 'cover']
    chunks = {p: [] for p in roles}
    counts = {'games': 0, 'bid_decisions': 0}
    observed_rewards = {'bid': [], 'cover': []}
    recorded_inputs = {'bid': [], 'cover': []}
    last_inputs = {}
    class StopAudit(BaseException):
        pass
    class RecordingEnvironment(Environment):
        def initial(self, *args, **kwargs):
            ret = super().initial(*args, **kwargs)
            self.previous = ret
            return ret
        def step(self, action):
            pos, obs, extra = self.previous
            if pos == 'bid':
                counts['bid_decisions'] += 1
                candidate = next(a for a in extra['legal_actions'] if a[0] == action[0])
                last_inputs['bid'] = torch.cat((torch.from_numpy(cards2matrix(candidate[1])), obs['x']), 0).to(torch.int8)
            elif pos == 'cover':
                last_inputs['cover'] = obs['x'].clone()
            ret = super().step(action)
            if 'game_reward' in ret[2]:
                counts['games'] += 1
                for p in observed_rewards:
                    # Unique synthetic rewards make the temporal shift unambiguous.
                    reward = counts['games'] / 10
                    ret[2]['game_reward'][p] = reward
                    observed_rewards[p].append(reward)
                    recorded_inputs[p].append(last_inputs[p])
            self.previous = ret
            return ret
    class SamplingModel:
        def bid(self, *args, **kwargs):
            return {'action': torch.tensor(0)}
        def cover(self, *args, **kwargs):
            return {'action': torch.ones(1, 120)}
        def play(self, role, zs, xs, z, x, legal_actions, legal_types, **kwargs):
            return {'action_type_index': legal_types.index(__SINGLE__) if __SINGLE__ in legal_types else 0,
                    'action': 0, 'strategy': F.one_hot(torch.tensor(0), 7).float()}
    class RecordingQueue:
        def __init__(self, role):
            self.role = role
        def put(self, chunk):
            if self.role in observed_rewards:
                chunks[self.role].append(chunk)
            if all(chunks[p] for p in observed_rewards):
                raise StopAudit()
    flags = SimpleNamespace(unroll_length=4, training_device='cpu', exp_epsilon=0., objective='adp')
    counter = lambda: SimpleNamespace(value=0, get_lock=threading.Lock)
    with patch.object(act_module, 'Environment', RecordingEnvironment):
        try:
            act_module.act(0, 'cpu', {p: RecordingQueue(p) for p in roles}, SamplingModel(),
                           counter(), counter(), SimpleNamespace(value=0.), flags)
        except StopAudit:
            pass
    result = dict(counts)
    for p in observed_rewards:
        chunk = chunks[p][0]
        result[p] = {'expected_rewards_for_inputs': observed_rewards[p][:4],
                     'actual_targets': chunk['target_wp'].tolist(),
                     'inputs_are_games_1_to_4': all(torch.equal(chunk['obs_x'][i], recorded_inputs[p][i]) for i in range(4))}
    return result


def environment_evidence():
    env = Env()
    obs = env.reset()
    while obs['position'] == 'bid':
        obs, _ = env.step([obs['legal_actions'][0][0]])
    game = env._env
    before = [set(game.getPlayerHandCards(i)) for i in range(4)]
    obs, _ = env.step([game.getPlayerHandCards(game.getBanker())[:8]])
    info = env.infoset
    result = {'first_play_mask_recovers_hidden_hands': {}, 'unknown_bottom_recovered': False}
    for i, role in enumerate(__PLAY_ROLES__):
        if role != 'banker':
            recovered = set(range(108)) - set(info.mask_cards[role])
            result['first_play_mask_recovers_hidden_hands'][role] = recovered == before[(game.getBanker() + i) % 4]
    # Advance one action to observe a nonbanker: all other hands + own hand reveal bottom.
    tp = obs['legal_types'][0]
    obs, _ = env.step([matrix2cards(obs['legal_actions'][tp][0]), tp])
    info = env.infoset
    known = set(info.player_hand_cards) | set(info.other_hand_cards)
    known |= {c for cards in info.played_cards.values() for c in cards}
    result['unknown_bottom_recovered'] = set(range(108)) - known == set(game.getPublicCards())
    # Reproduce evaluator's cached infoset after roundend -> play.
    stale_trials = stale_seat = 0
    discard_example = None
    games, moves, seen = 0, 0, Counter()
    while games < 20:
        if obs['position'] == 'bid':
            action = [obs['legal_actions'][random.randrange(len(obs['legal_actions']))][0]]
        elif obs['position'] == 'cover':
            action = [random.sample(env.infoset.player_hand_cards, 8)]
        else:
            tp = random.choice(obs['legal_types'])
            ai = random.randrange(len(obs['legal_actions'][tp]))
            action = [matrix2cards(obs['legal_actions'][tp][ai]), tp]
            seen[tp] += 1
            moves += 1
        old_masks = copy.deepcopy(game.mask_cards)
        acting_seat = game.getPlayerPosition()
        old_stage = game.getStage()
        obs, done = env.step(action)
        if old_stage == 'play' and action[-1] == 5 and game.getStage() == 'play' and discard_example is None:
            changed = [r for r in __PLAY_ROLES__ if old_masks[r] != game.mask_cards[r]]
            if changed:
                discard_example = {'acting_role': game._role_of(acting_seat, game.getBanker()),
                                   'changed_masks': changed, 'next_role': env.infoset.player_position}
        if done:
            rewards = env._get_reward()
            assert rewards['banker'] == rewards['banker_op'] == -rewards['banker_up'] == -rewards['banker_down']
            assert (rewards['banker'] > 0) == (env._get_game_score() < 80)
            games += 1
            obs = env.reset()
        elif env._stage == 'roundend':
            stale = env.infoset
            obs, _ = env.step(action)
            stale_trials += 1
            stale_seat += stale.seat != env.infoset.seat
        # Card conservation and matrix encoding across ordinary valid play.
        if game.getStage() == 'play':
            cards = sum([game.getPlayerHandCards(i) for i in range(4)], [])
            cards += sum([game.getPlayedCards(i) for i in range(4)], []) + game.getPublicCards()
            assert sorted(cards) == list(range(108))
    result.update({'completed_games': games, 'random_play_actions': moves, 'action_types': dict(seen),
                   'stale_infoset_after_roundend': {'trials': stale_trials, 'different_seat': stale_seat},
                   'discard_updates_wrong_role': discard_example})
    # Reward range: a single-card last trick can exceed 80 points including bottom.
    from rlcard.games.tractors.env.botzone import tractorGame
    e = Env()
    g = e._env
    g.setMajor('s', '2')
    g.globalInfo = {'stage': 'play', 'level': '2', 'game_score': 0,
                    'banking': {'banker': 0, 'banker_last': 0, 'major': 's'},
                    'publiccard': [36, 37, 38, 39, 48, 49, 50, 51],
                    'history': [[], [], 0, 0], 'playerpos': 0}
    g.player_hand_cards = [[9], [13], [17], [21]]
    for seat in range(4):
        tractorGame.step(g, [seat, [g.player_hand_cards[seat][0]]])
    result['last_trick_reward_over_one'] = e._get_step_reward()
    return result


def engine_edge_evidence():
    from rlcard.games.tractors.env.botzone import tractorGame
    result = {}
    g = tractorGame()
    g.setMajor('s', '2')
    g.globalInfo = {'banking': {'major': 's'}, 'stage': 'play'}
    lead, hand = [9, 63, 13, 67], [21, 8, 62, 12, 66]
    actions = g.getLegalPlayCard([lead], hand, '2')
    candidate = next(a for aa in actions for a in aa)
    g.checkLegalMove([g.num2Poker(c) for c in candidate], '2', 's', 1, [lead], [lead, hand, [], []], 0)
    result['ruff_with_lead_suit_still_in_hand'] = {
        'lead': [g.num2Poker(c) for c in lead], 'hand': [g.num2Poker(c) for c in hand],
        'generated': [g.num2Poker(c) for c in candidate], 'validator_error': g.getErrorCode(),
    }
    g = tractorGame()
    g.setMajor('s', '2')
    g.globalInfo = {'stage': 'play', 'level': '2', 'game_score': 0,
                    'banking': {'banker': 0, 'banker_last': 0, 'major': 's'},
                    'publiccard': [], 'history': [[], [], 0, 0], 'playerpos': 0}
    g.player_hand_cards = [[9, 17, 8], [21, 12], [13, 20], [25, 24]]
    g.step([0, [9, 17]])
    penalty = {'immediately_after_throw': g.getLastRoundScore(), 'score_after_throw': g.getGameScore()}
    for seat, card in [(1, 21), (2, 13), (3, 25)]:
        g.step([seat, [card]])
    penalty.update(final_trick_reward=g.getLastRoundScore(), final_game_score=g.getGameScore())
    result['failed_throw_penalty_lost'] = penalty
    # Follow actual engine transitions across 300 complete games, with legal singles.
    g = tractorGame()
    g.reset()
    levels, finalends = [], []
    levels.append(g.getLevel())
    for _ in range(100000):
        stage, seat = g.getStage(), g.getPlayerPosition()
        if stage == 'bid':
            g.step([seat, random.choice(g.getLegalBidActions(seat))])
        elif stage == 'cover':
            g.step([seat, random.sample(g.getPlayerHandCards(seat), 8)])
        elif stage == 'roundend':
            g.step(None)
        elif stage == 'play':
            hand = g.getPlayerHandCards(seat)
            history = g.getCurrRoundPlayHistory()
            if history:
                first = history[0][0]
                major_cards = set(g.getMajorCards())
                same = [c for c in hand if (c in major_cards if first in major_cards
                                            else c not in major_cards and (c % 54) % 4 == (first % 54) % 4)]
                if same:
                    hand = same
            g.step([seat, [random.choice(hand)]])
        elif stage in ('gameend', 'finalend'):
            if stage == 'finalend':
                finalends.append(len(levels))
            if len(levels) >= 300:
                break
            g.reset()
            levels.append(g.getLevel())
        else:
            raise AssertionError(stage)
        assert g.getErrorCode() == 0
    assert len(levels) == 300 and finalends
    result['series_reset'] = {'games': len(levels), 'first_finalend_game': finalends[0],
        'finalend_count': len(finalends), 'first_30_levels': levels[:30],
        'last_100_level_counts': dict(Counter(levels[-100:])),
        'level_2_count_after_first_series': levels[finalends[0]:].count('2')}
    return result


def main():
    random.seed(20260920)
    np.random.seed(20260920)
    torch.manual_seed(20260920)
    torch.set_num_threads(2)
    result = {}
    # The engine uses unseeded default_rng() internally; make this audit reproducible.
    original_default_rng = np.random.default_rng
    audit_rng = original_default_rng(20260920)
    with patch.object(np.random, 'default_rng', side_effect=lambda seed=None: audit_rng if seed is None else original_default_rng(seed)):
        for name, fn in [('logs', log_evidence), ('model', model_evidence),
                         ('actor', actor_alignment_evidence), ('environment', environment_evidence),
                         ('engine_edges', engine_edge_evidence)]:
            result[name] = fn()
            print(name, json.dumps(result[name], ensure_ascii=True), flush=True)
    out = ROOT / 'docs/audits/2026-09-20-evidence.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
