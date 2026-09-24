import random

import numpy as np
import torch
from collections import Counter

from .agent import Agent
from ..env.utils import (
    __CARDSCALE__, __SUITSET__, __POINT__, __MAJOR__,
    __SINGLE__, __PAIR__, __TRACTOR__, __SUSPECT__,
    __PLAYER_COUNT__, __PLAY_ROLES__,
    matrix2cards,
)

# 分牌对应的点数索引（__CARDSCALE__ = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']）
# 5 / 10(0) / K 是计分牌
_SCORE_RANKS = {4, 7, 12}


def _to_cards(mat):
    """将 (2,4,15) 的牌矩阵还原为牌id列表"""
    if hasattr(mat, 'cpu'):
        mat = mat.cpu().numpy()
    return matrix2cards(mat).tolist()


def _card_strength(card):
    """粗略估算单张牌的强度（未考虑主牌/级牌，作为简单基线）"""
    m = card % 54
    if m == 52:  # 小王
        return 14
    if m == 53:  # 大王
        return 15
    r = m // 4
    if r == 0:  # A
        return 12
    if r == 1:  # 2
        return 13
    return r - 1  # 3..K -> 1..11


def _is_score(card):
    return (card % 54) // 4 in _SCORE_RANKS


def _score_points(card):
    """单张牌的计分数值：5->5分，10/K->10分，其余0分。"""
    r = (card % 54) // 4
    if r == 4:      # 5
        return 5
    if r == 7:      # 10
        return 10
    if r == 12:     # K
        return 10
    return 0


def _poker_of(card):
    """牌id -> 扑克字符串（如 's6'、'c0'、'jo'、'Jo'）。"""
    m = card % 54
    if m == 52:
        return 'jo'
    if m == 53:
        return 'Jo'
    return __SUITSET__[m % 4] + __CARDSCALE__[m // 4]


def _trump_order(major, level):
    """重建主牌顺序（弱->强），返回 poker_str -> 排名 的映射。

    与 botzone.setMajor 保持一致：主花色非级牌（2..A，弱到强）、
    其它花色级牌、主花色级牌，最后是小王/大王。
    """
    if major in (None, '', 'n'):
        order = ([s + level for s in __SUITSET__] if major == 'n' else []) + list(__MAJOR__)
    else:
        order = (
            [major + p for p in __POINT__ if p != level]
            + [s + level for s in __SUITSET__ if s != major]
            + [major + level]
            + list(__MAJOR__)
        )
    return {p: i for i, p in enumerate(order)}


# 复刻 cards2matrix 的填充逻辑（插入顺序必须一致），得到展平后每个位置对应的牌id（-1为填充位）
_CARD_GRID = np.arange(108, dtype=np.int64)
_CARD_GRID = np.insert(_CARD_GRID, 53, [-1, -1, -1])
_CARD_GRID = np.insert(_CARD_GRID, 57, [-1, -1, -1])
_CARD_GRID = np.insert(_CARD_GRID, 60 + 53, [-1, -1, -1])
_CARD_GRID = np.insert(_CARD_GRID, 60 + 57, [-1, -1, -1])
_CARD_GRID = np.transpose(_CARD_GRID.reshape(2, 15, 4), (0, 2, 1)).flatten()
_CARD_POS = {int(c): i for i, c in enumerate(_CARD_GRID) if c >= 0}


def _check_type(pokers, level):
    """复刻 checkPokerType，返回牌型：单牌/对子/拖拉机/其它(甩牌或贴牌)。"""
    if len(pokers) == 1:
        return __SINGLE__
    if len(pokers) == 2:
        return __PAIR__ if pokers[0] == pokers[1] else __SUSPECT__
    if len(pokers) % 2 == 0:
        c = Counter(pokers)
        if 'jo' in c and 'Jo' in c and c['jo'] == 2 and c['Jo'] == 2 and len(pokers) == 4:
            return __TRACTOR__
        if 'jo' in c or 'Jo' in c:
            return __SUSPECT__
        for v in c.values():
            if v != 2:
                return __SUSPECT__
        suit = next(iter(c))[0]
        pts = []
        for k in c:
            if k[0] != suit or k[1] == level:
                return __SUSPECT__
            pts.append(__POINT__.index(k[1]))
        pts.sort()
        for i in range(len(pts) - 1):
            if pts[i + 1] - pts[i] != 1:
                return __SUSPECT__
        return __TRACTOR__
    return __SUSPECT__


def _max_rank(pokers, trump_order):
    """一组牌的最大强度：主牌(1000+主序)恒大于副牌(点数序)。"""
    best = -1
    for p in pokers:
        if p in trump_order:
            r = 1000 + trump_order[p]
        else:
            r = __POINT__.index(p[1])
        if r > best:
            best = r
    return best


def _beats(challenger, winner, lead, trump_order, level):
    """challenger 是否能压过当前 winner（依据首出 lead 的牌型）。

    复刻 getWinner 中“常规牌型”的比较逻辑：
    牌型不一致 / 花色不对 / 副牌压不过主牌 => 视为贴牌，无法压过。
    """
    lead_typ = _check_type(lead, level)
    if _check_type(challenger, level) != lead_typ:
        return False  # 牌型不符，视为贴牌

    lead_trump = lead[0] in trump_order
    chal_trump = challenger[0] in trump_order
    win_trump = winner[0] in trump_order

    if lead_trump:
        if not chal_trump:
            return False
        if not win_trump:
            return True
        return _max_rank(challenger, trump_order) > _max_rank(winner, trump_order)

    # 首出为非主牌花色
    lead_suit = lead[0][0]
    if chal_trump:
        if win_trump:
            return _max_rank(challenger, trump_order) > _max_rank(winner, trump_order)
        return True  # 主牌压副牌
    if challenger[0][0] != lead_suit:
        return False  # 花色不对
    if win_trump:
        return False  # 当前胜者是主牌
    if winner[0][0] != lead_suit:
        return True  # 当前胜者是贴牌，同花色即可压过
    return _max_rank(challenger, trump_order) > _max_rank(winner, trump_order)


def _trick_winner_seat(round_play_cards, lead_seat, trump_order, level):
    """在当前轮已出的牌中，判定当前牌权归属的座位(0-3)。"""
    winner_seat = lead_seat
    winner_cards = round_play_cards[__PLAY_ROLES__[lead_seat]]
    lead_pokers = [_poker_of(c) for c in winner_cards]

    for off in range(1, __PLAYER_COUNT__):
        seat = (lead_seat + off) % __PLAYER_COUNT__
        cards = round_play_cards.get(__PLAY_ROLES__[seat], [])
        if not cards:
            continue
        chal = [_poker_of(c) for c in cards]
        win = [_poker_of(c) for c in winner_cards]
        if _beats(chal, win, lead_pokers, trump_order, level):
            winner_seat = seat
            winner_cards = cards
    return winner_seat


# 基于规则的Agent
class RuleAgent(Agent):
    """基于规则的Agent，作为评估基线。

    play 依据局面信息（当前轮已出的牌、牌权归属、队友是否大起来）选择出牌：
    - 首出：优先出最有把握大起来的牌（拖拉机/对子/单A），否则随机出一张非分单牌。
    - 跟牌且队友当前最大：垫分牌（5/10/K）进分。
    - 跟牌且对手当前最大：能用最小的牌压过则压，否则垫最弱非分牌。
    """

    def _lead_action(self, plays, trump_order, level):
        """首出：优先出最有把握大起来的牌（拖拉机/对子/单A），否则随机出一张非分单牌。"""
        tractor = None
        pair = None
        single_a = None
        non_score_singles = []

        for type_i, act_i, cards, pokers in plays:
            typ = _check_type(pokers, level)
            strength = _max_rank(pokers, trump_order)
            if typ == __TRACTOR__:
                if tractor is None or strength > tractor[0]:
                    tractor = (strength, type_i, act_i)
            elif typ == __PAIR__:
                if pair is None or strength > pair[0]:
                    pair = (strength, type_i, act_i)
            elif typ == __SINGLE__:
                if pokers[0][1] == 'A':
                    if single_a is None or strength > single_a[0]:
                        single_a = (strength, type_i, act_i)
                if not _is_score(cards[0]):
                    non_score_singles.append((type_i, act_i))

        if tractor is not None:
            return dict(action_type_index=tractor[1], action=tractor[2])
        if pair is not None:
            return dict(action_type_index=pair[1], action=pair[2])
        if single_a is not None:
            return dict(action_type_index=single_a[1], action=single_a[2])
        if non_score_singles:
            type_i, act_i = random.choice(non_score_singles)
            return dict(action_type_index=type_i, action=act_i)

        # 兜底：没有非分单牌时随机出一张单牌（理论上极少发生）
        singles = [(ti, ai) for ti, ai, _cards, pokers in plays if _check_type(pokers, level) == __SINGLE__]
        if singles:
            ti, ai = random.choice(singles)
            return dict(action_type_index=ti, action=ai)
        return dict(action_type_index=plays[0][0], action=plays[0][1])

    def play(self, position, z, x, legal_actions, legal_card_type, infosets, flags=None):
        major = getattr(infosets, 'major', '')
        level = getattr(infosets, 'level', '2')
        trump_order = _trump_order(major, level)

        seat = infosets.seat
        round_play_cards = infosets.round_play_cards or {}
        lead_seat = infosets.round_play_lead_seat
        if lead_seat is None:
            lead_seat = seat
        is_lead = (lead_seat == seat)

        # 收集所有合法动作及其牌面
        plays = []
        for i, tp in enumerate(legal_card_type):
            acts = legal_actions[tp]
            for a in range(len(acts)):
                cards = _to_cards(acts[a])
                pokers = [_poker_of(c) for c in cards]
                plays.append((i, a, cards, pokers))

        if is_lead:
            return self._lead_action(plays, trump_order, level)

        winner_seat = _trick_winner_seat(round_play_cards, lead_seat, trump_order, level)
        teammate_win = (winner_seat % 2 == seat % 2)
        winner_pokers = [_poker_of(c) for c in round_play_cards[__PLAY_ROLES__[winner_seat]]]
        lead_pokers = [_poker_of(c) for c in round_play_cards[__PLAY_ROLES__[lead_seat]]]

        best = None
        for p in plays:
            type_i, act_i, cards, pokers = p
            strength = _max_rank(pokers, trump_order)

            if teammate_win:
                # 队友已最大：优先多垫分牌（负分越小越好），其次垫弱牌
                cost = (-sum(_score_points(c) for c in cards), strength)
            elif _beats(pokers, winner_pokers, lead_pokers, trump_order, level):
                # 能压过对手：用最弱的牌压
                cost = (0, strength)
            else:
                # 压不过：垫最弱、尽量非分牌
                cost = (sum(_score_points(c) for c in cards), strength)

            if best is None or cost < best[0]:
                best = (cost, (type_i, act_i))

        return dict(action_type_index=best[1][0], action=best[1][1])

    def cover(self, z, x, infosets, flags=None):
        # 埋掉8张最弱且尽量非分牌的牌：优先埋强度低、非分牌
        hand = _to_cards(x[0, :2])
        order = sorted(hand, key=lambda c: (_card_strength(c), 1 if _is_score(c) else 0))
        bury = order[:8]

        logits = np.zeros(2 * 4 * 15, dtype=np.float32)
        for c in bury:
            logits[_CARD_POS[c]] = 1.0
        action = torch.from_numpy(logits).unsqueeze(0)
        return dict(action=action)

    def bid(self, z, x, infosets, flags=None):
        # x: (num_actions, 10, 4, 15)，其中 [:2] 为当前动作的主牌矩阵，[2:4] 为手牌矩阵
        hand = _to_cards(x[0, 2:4])
        suit_cnt = [0, 0, 0, 0]
        for c in hand:
            m = c % 54
            if m < 52:  # 排除大小王
                suit_cnt[m % 4] += 1

        # 在手牌最多的花色中叫主，若都不值得叫则保持“不叫”（首项）
        best_idx = 0
        best_cnt = 0
        for i in range(x.shape[0]):
            suit = None
            for c in _to_cards(x[i, :2]):
                m = c % 54
                if m < 52:
                    suit = m % 4
                    break
            if suit is None:
                continue  # 不叫/无主，跳过
            if suit_cnt[suit] > best_cnt:
                best_cnt = suit_cnt[suit]
                best_idx = i

        action = torch.tensor([best_idx])
        return dict(action=action)