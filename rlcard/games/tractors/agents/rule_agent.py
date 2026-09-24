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

    出牌策略融合了拖拉机进阶玩法中的几个维度：
    - 首出：优先长拖拉机（长拖压制）、其次甩牌、对子、单A、非分单牌。
    - 跟牌：队友最大时垫分；对手最大时用最小牌压；压不过则垫最弱且尽量不拆对。
    - 叫主：按主牌强度（花色张数 + 王 + 是否持级牌对）叫主。
    - 扣底：保留主牌与副牌A用于控场，先埋非分小牌，主牌强时可埋分牌。
    """

    @staticmethod
    def _lead_key(tp, cards, pokers, trump_order):
        """首出动作的优先级（数值越大越优先）。"""
        p0 = pokers[0]
        is_trump = int(p0 in trump_order)
        strength = _max_rank(pokers, trump_order)
        n = len(pokers)
        if tp == __TRACTOR__:
            # 长拖优先（对子数目更多越难被压）；主拖优先于副拖
            return (5, is_trump, n // 2, strength)
        if tp == __SUSPECT__:
            # 甩牌：同为“纯大”时优先甩更多张；主牌甩牌更稳
            return (4, is_trump, n, strength)
        if tp == __PAIR__:
            return (3, is_trump, int(p0[1] == 'A'), strength)
        # 单牌
        score = _score_points(cards[0])
        if p0[1] == 'A':
            return (2, is_trump, 0, strength)
        if score == 0:
            # 非分单牌优先；同类别下副牌单牌优先于主牌单牌（保留主牌控场）
            return (1, is_trump, strength, 0)
        return (0, is_trump, strength, 0)

    def _lead_action(self, plays, trump_order):
        best = None
        best_key = None
        for tp, type_i, act_i, cards, pokers in plays:
            key = self._lead_key(tp, cards, pokers, trump_order)
            if best_key is None or key > best_key:
                best_key = key
                best = (type_i, act_i)
        return dict(action_type_index=best[0], action=best[1])

    def play(self, position, z_strategy, x_strategy, z, x, legal_actions, legal_card_type, infosets, flags=None):
        # 规则Agent不使用状态级表示 z_strategy/x_strategy
        major = getattr(infosets, 'major', '')
        level = getattr(infosets, 'level', '2')
        trump_order = _trump_order(major, level)
        hand = getattr(infosets, 'player_hand_cards', None) or []

        seat = infosets.seat
        round_play_cards = infosets.round_play_cards or {}
        lead_seat = infosets.round_play_lead_seat
        if lead_seat is None:
            lead_seat = seat
        is_lead = (lead_seat == seat)

        # 收集所有合法动作及其牌面、真实牌型
        plays = []
        for i, tp in enumerate(legal_card_type):
            acts = legal_actions[tp]
            for a in range(len(acts)):
                cards = _to_cards(acts[a])
                pokers = [_poker_of(c) for c in cards]
                plays.append((tp, i, a, cards, pokers))

        if is_lead:
            return self._lead_action(plays, trump_order)

        winner_seat = _trick_winner_seat(round_play_cards, lead_seat, trump_order, level)
        teammate_win = (winner_seat % 2 == seat % 2)
        winner_pokers = [_poker_of(c) for c in round_play_cards[__PLAY_ROLES__[winner_seat]]]
        lead_pokers = [_poker_of(c) for c in round_play_cards[__PLAY_ROLES__[lead_seat]]]

        # 手牌中的对子集合：跟牌垫牌时尽量不拆对
        hand_counter = Counter(_poker_of(c) for c in hand)
        hand_paired = {p for p, v in hand_counter.items() if v >= 2}

        best = None
        for tp, type_i, act_i, cards, pokers in plays:
            strength = _max_rank(pokers, trump_order)
            if teammate_win:
                # 队友已最大：优先垫分牌进分，其次不拆对、垫弱牌
                cost = (-sum(_score_points(c) for c in cards),
                        int(any(_poker_of(c) in hand_paired for c in cards)),
                        strength)
            elif _beats(pokers, winner_pokers, lead_pokers, trump_order, level):
                # 能压过对手：用最弱的牌压
                cost = (0, 0, strength)
            else:
                # 压不过：垫最弱、尽量非分牌，且不拆对
                cost = (sum(_score_points(c) for c in cards),
                        int(any(_poker_of(c) in hand_paired for c in cards)),
                        strength)

            if best is None or cost < best[0]:
                best = (cost, (type_i, act_i))

        return dict(action_type_index=best[1][0], action=best[1][1])

    def cover(self, z, x, infosets, flags=None):
        """扣底：保留主牌与副牌A控场，先埋非分小牌；主牌强(>=8)时可埋分牌。"""
        hand = _to_cards(x[0, :2])
        major = getattr(infosets, 'major', '')
        level = getattr(infosets, 'level', '2')
        trump_order = _trump_order(major, level)
        n_trump = sum(1 for c in hand if _poker_of(c) in trump_order)

        def keep_value(c):
            p = _poker_of(c)
            if p in trump_order:
                if p in ('jo', 'Jo'):
                    return 1000  # 大小王必须保留
                return 600 + trump_order[p]  # 主牌保留用于控场，越大越留
            if p[1] == 'A':
                return 500  # 副牌A保留控制一门花色
            if _score_points(c) > 0:
                # 分牌：主牌充足时埋底保护，否则留在手中
                return 200 if n_trump >= 8 else 400
            return _card_strength(c)  # 非分副牌，越小越先埋

        bury = sorted(hand, key=keep_value)[:8]

        logits = np.zeros(2 * 4 * 15, dtype=np.float32)
        for c in bury:
            logits[_CARD_POS[c]] = 1.0
        action = torch.from_numpy(logits).unsqueeze(0)
        return dict(action=action)

    def bid(self, z, x, infosets, flags=None):
        """叫主：按主牌强度（花色张数 + 王 + 是否持级牌对）选择，无更好选择则“不叫”。"""
        hand = _to_cards(x[0, 2:4])
        hand_set = set(hand)
        level = getattr(infosets, 'level', '2')
        legal = getattr(infosets, 'legal_actions', None) or []

        suit_cnt = [0, 0, 0, 0]
        for c in hand:
            m = c % 54
            if m < 52:  # 排除大小王
                suit_cnt[m % 4] += 1
        joker_cnt = sum(1 for c in hand if c % 54 >= 52)

        level_int = __CARDSCALE__.index(level) * 4

        def bid_score(idx):
            if idx >= len(legal):
                return float('-inf')
            act = legal[idx][0]
            if act in (None, ''):
                return -1.0  # “不叫”作为最低基线
            if act == 'n':
                # 无主：仅有级牌与王为主牌，能被叫出说明持双王，强度可观
                return 6 + joker_cnt
            sidx = __SUITSET__.index(act)
            score = suit_cnt[sidx] + joker_cnt  # 该花色普通牌 + 王 均成为主牌
            if level_int + sidx in hand_set and level_int + sidx + 54 in hand_set:
                score += 2  # 持有该花色级牌对，可加固/反主
            return score

        best_idx = 0
        best_score = bid_score(0)
        for i in range(1, len(legal)):
            s = bid_score(i)
            if s > best_score:
                best_score = s
                best_idx = i

        action = torch.tensor([best_idx])
        return dict(action=action)