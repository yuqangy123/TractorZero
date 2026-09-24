from collections import Counter
import numpy as np

from .game import GameEnv
from .utils import *



deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

# 分数牌（分牌）：5/10/K 两张副
_SCORE_POKES = frozenset(
    [16, 17, 18, 19, 36, 37, 38, 39, 48, 49, 50, 51]
    + [c + 54 for c in (16, 17, 18, 19, 36, 37, 38, 39, 48, 49, 50, 51)]
)


class Env:

    def __init__(self, objective=None):
        self._objective = objective

        # Initialize the internal environment
        self._env = GameEnv()
        self.total_round = 0
        self.infoset = None

    def reset(self):
        self._env.reset()

        self.infoset = self._bid_infoset
        return get_obs(self.infoset, self._stage)

    def step(self, action):
        # if self._bid_over:
        #     if self._cover_over:
        #        pos = self._acting_player_position
        #     else:    
        #         pos = self._coving_player_position
        # else:
        #     pos = self._bidding_player_position

        self._env.step(action)
        
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            # reward = {
            #     "play": {
            #         role: self._get_reward(role) for role in __PLAY_ROLES__
            #     }
            # }
            obs = None
            
        else:
            if self._bid_over:
                if self._cover_over:
                    self.infoset = self._game_infoset
                else:    
                    self.infoset = self._cover_infoset
            else:
                self.infoset = self._bid_infoset
            obs = get_obs(self.infoset, self._stage)
            # reward = {
            #     "play": {
            #         "banker": self._get_reward("banker"),
            #         "banker_down": self._get_reward("banker_down"),
            #         "banker_up": self._get_reward("banker_up"),
            #     }
            # }
        return obs, done

    def _get_step_reward(self):
        round_win_score = self._env.getLastRoundScore()
        bid_score = self._env.getLeastBidScore()
        
        # r = round_score/100.0#4*K + 4*10 + 4*5
        # if 0.0 < round_score:
        #     if game_score <= bid_score:
        #         up_level_score = (game_score/bid_score) ** 1.3
        #     else:
        #         up_level_score = (game_score/200.) ** 0.7
                
        #     up_level_reward += up_level_score*0.5
        # else:
        #     r = -self._env.getPlayerLeftHandCards(banker)/__HAND_CARD_NUM__ * 0.5
        r = 0.
        if round_win_score > 0:
            r = round_win_score/bid_score
            reward = {'banker': -r,
            'banker_op': -r,
            'banker_up': r,
            'banker_down': r}

        else:
            play_seq = self._env.getLastRoundPlayHistory()
            action_score = 0
            for action in play_seq:
                for c in action:
                    if c in __SCORE__CARD__[:8]:action_score += 5
                    elif c in __SCORE__CARD__[8:]:action_score += 10
            r = action_score/bid_score
            reward = {'banker': r,
                        'banker_op': r,
                        'banker_up': -r,
                        'banker_down': -r}
        
        return reward

    def _get_reward(self):
        banker = self._env.getLastBanker()
        # print('getLastBanker:', banker)#todo
        
        bid_score = self._env.getLeastBidScore()#叫分
        game_score = self._env.getGameScore()#局内得分
        
        public_score = 0
        #[16, 17, 18, 19, 36, 37, 38, 39, 48, 49, 50, 51]
        for c in self._env.getPublicCards():
            if c in [16, 17, 18, 19, 16+54, 17+54, 18+54, 19+54]:public_score += 5
            elif c in [36, 37, 38, 39, 36+54, 37+54, 38+54, 39+54]:public_score += 10
            elif c in [48, 49, 50, 51, 48+54, 49+54, 50+54, 51+54]:public_score += 10
        
        #庄家赢了则符号为正，否则为负
        banker_win = 1. if bid_score > game_score else -1.
        
        reward = {}
        #bid和cover都是以banker为actor
        reward['cover_public_score'] = public_score/100. * banker_win
        reward['cover'] = banker_win
        
        if bid_score == game_score:
            reward['bid'] = 5./bid_score
        else:
            bid_diff_ratio = (bid_score - game_score) / bid_score
            reward['bid'] = np.clip(bid_diff_ratio, -1.0, 1.0)
        
        # print('end_score:', end_score)#todo
        reward['banker'] = self._env.getEndingScore(banker)
        reward['banker_down'] = self._env.getEndingScore((banker+1)%__PLAYER_COUNT__)
        reward['banker_op'] = self._env.getEndingScore((banker+2)%__PLAYER_COUNT__)        
        reward['banker_up'] = self._env.getEndingScore((banker+3)%__PLAYER_COUNT__)
        return reward
    
    def _get_infosets(self):
        return self.infoset
    
    #局内得分
    def _get_game_score(self):
        return self._env.getGameScore()
    
    #当前回合捡分
    def _get_round_score(self):
        return self._env.getLastRoundScore()
    
    #上一轮被打出的分牌（5/10/K）
    def _get_round_score_poke(self):
        pokes = []
        for play in self._env.getLastRoundPlayHistory():
            for c in play:
                if c in _SCORE_POKES:
                    pokes.append(c)
        return pokes
    
    def _get_round_banker(self):
        # EndGame 已为下一局换庄，末墩的角色仍须按刚结束的本局庄家映射。
        if self._env.getStage() in ('gameend', 'finalend'):
            return self._env.getLastBanker()
        return self._env.getBanker()

    #上一轮夺得牌权的角色（banker/banker_op/banker_down/banker_up）
    def _get_last_winner_role(self):
        return self._env._role_of(self._env.getLastRoundWinSeat(), self._get_round_banker())
    
    #上一轮首出的角色（谁先出的这张牌）
    def _get_round_lead_role(self):
        return self._env._role_of(self._env.getLastRoundPlaySeat(), self._get_round_banker())
    
    #是否为本局最后一墩：本墩结束后已有人手牌出完
    def _get_is_last_round(self):
        banker = self._env.getBanker()
        remain = [self._env.getPlayerLeftHandCards((banker + i) % __PLAYER_COUNT__)
                  for i in range(__PLAYER_COUNT__)]
        return 1.0 if min(remain) == 0 else 0.0

    #冻结刚结束那一墩的元数据与各角色的实际动作特征，供策略标签使用。
    #必须在环境 reset 之前调用：reset 后旧墩的赢家/得分会被新一局覆盖，is_last_round 也会失真。
    def _get_step_meta(self):
        env = self._env
        banker = self._get_round_banker()
        winner_seat = env.getLastRoundWinSeat()
        lead_seat = env.getLastRoundPlaySeat()
        majors = set(env.getMajorCards())
        # play_seq 每次出牌追加一条 [seat, cards, type]，最后 __PLAYER_COUNT__ 条即刚结束的一墩
        last_round = env.getPlayHistory()[-__PLAYER_COUNT__:]

        def _group(card):
            # 同一「跟门」组：主牌归为一组，副牌按花色分组
            return 'M' if card in majors else (card % 54) % 4

        lead_group = _group(last_round[0][1][0]) if last_round and last_round[0][1] else None
        roles = {}
        for seat, cards, card_type in last_round:
            roles[env._role_of(seat, banker)] = {
                'is_lead': float(seat == lead_seat),
                'threw': float(card_type == __SUSPECT__),
                'followed': float(any(_group(c) == lead_group for c in cards))
                            if lead_group is not None else 1.0,
                'won': float(seat == winner_seat),
                'score_in_play': float(any(c in _SCORE_POKES for c in cards)),
            }
        return {
            'roles': roles,
            'winner_role': env._role_of(winner_seat, banker),
            'round_score': float(env.getLastRoundScore()),
            'score_played': float(len(self._get_round_score_poke()) > 0),
            'is_last_round': float(self._get_is_last_round()),
        }

    def _get_last_bid_rule(self):
        return self._env._role_of(self._env.getLastBidSeat(), self._env.getLastBanker())
    
    def print_game_log(self, b):
        self._env.print_game_log(b)
        
    @property
    def _game_infoset(self):
        return self._env.game_infoset[self._env.player_rule]

    @property
    def _bid_infoset(self):
        return self._env.bid_infoset
    
    @property
    def _cover_infoset(self):
        return self._env.cover_infoset

    @property
    def _coving_player_position(self):
        return self._env.coving_player_position
    
    @property
    def _acting_player_position(self):
        return self._env.acting_player_position

    @property
    def _bidding_player_position(self):
        return self._env.bidding_player_position

    @property
    def _game_over(self):
        return self._env.game_over

    @property
    def _bid_over(self):
        return self._env.bid_over
    
    @property
    def _cover_over(self):
        return self._env.cover_over
    
    @property
    def _major(self):
        return self._env.getMajorColor()
    
    @property
    def _level(self):
        return self._env.getLevel()
    
    @property
    def _stage(self):
        return self._env.stage

    @property
    def _game_winner(self):
        return self._env.get_winner()

    @property
    def _bid_winner(self):
        return self._env.get_winner_bid()



class DummyAgent(object):
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        self.action = action


def get_obs(infoset, stage):
    if infoset.player_position not in __PLAY_ROLES__ + ['bid', 'cover']:
        raise ValueError('')
    if stage == 'bid':
        return _get_bid_obs_resnet(infoset)
    elif stage == 'cover':
        return _get_cover_obs_resnet(infoset)
    else:
        return _get_play_obs_resnet(infoset)
        
    
def _get_one_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards)
    if num_left_cards > 0:
        one_hot[num_left_cards - 1] = 1

    return one_hot
def _get_one_hot_array_ex(num_left_cards, max_num_cards, one_value):
    one_hot = np.zeros(max_num_cards)
    one_hot[:] = one_value
    if num_left_cards > 0:
        one_hot[num_left_cards-1] = 1

    return one_hot

def _get_full_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards)
    if num_left_cards > 0:
        one_hot[:num_left_cards] = 1

    return one_hot

def _process_action_seq(sequence, length=10):
    sequence = sequence[-length:]
    mats = []
    seats = []
    for item in sequence:
        role, cards = item[0], item[1]
        mats.append(cards2matrix(cards))
        seats.append(_get_one_hot_array(role + 1, __PLAYER_COUNT__))
    mats = mats[::-1]
    seats = seats[::-1]
    if len(mats) < length:
        pad = length - len(mats)
        mats.extend([cards2matrix([]) for _ in range(pad)])
        seats.extend([np.zeros(__PLAYER_COUNT__, dtype=np.float32) for _ in range(pad)])
    action_seq = np.concatenate(mats, axis=0)   # (2*length, 4, 15)
    seats_seq = np.concatenate(seats, axis=0)  # (length*4,)
    return action_seq, seats_seq

def _get_play_obs_resnet(infoset):
    # num_legal_actions = len(infoset.legal_actions)
    # my_handcards_batch = np.repeat(my_handcards[np.newaxis, :,:,:],
    #                                num_legal_actions, axis=0)
    major = infoset.major
    level = infoset.level
    rule_index = __PLAY_ROLES__.index(infoset.player_position)
    
    my_handcards = cards2matrix(infoset.player_hand_cards)
    major_cards = cards2matrix(infoset.majorCards)
    
    played_cards = [cards2matrix(infoset.played_cards[p]) for p in __PLAY_ROLES__]
    played_cards = np.concatenate(played_cards, axis=0)
    
    last_round_play_cards = [cards2matrix(infoset.last_round_play_cards[p]) for p in __PLAY_ROLES__]
    last_round_play_cards = np.concatenate(last_round_play_cards, axis=0)
    
    round_play_cards = [cards2matrix(infoset.round_play_cards[__PLAY_ROLES__[(infoset.round_play_lead_seat+i)%__PLAYER_COUNT__]]) for i in range(__PLAYER_COUNT__-1)]
    round_play_cards = np.concatenate(round_play_cards, axis=0)
    
    other_handcards = cards2matrix(infoset.other_hand_cards)
    
    public_cards = cards2matrix(infoset.public_cards)
    
    remain_score_cards = cards2matrix(infoset.remain_score_cards)
    
    
    play_hand_mask_cards = [cards2matrix(infoset.mask_cards[p]) for p in __PLAY_ROLES__ if p != infoset.player_position]
    play_hand_mask_cards = np.concatenate(play_hand_mask_cards, axis=0)
    
    round_play_lead_seat = _get_one_hot_array(infoset.round_play_lead_seat+1, __PLAYER_COUNT__)
    
    #我的座位
    my_seat = _get_one_hot_array(infoset.seat+1, __PLAYER_COUNT__)
    #我的出牌顺序
    my_play_order = _get_one_hot_array(infoset.play_order+1, __PLAYER_COUNT__)
    
    
    #剩余牌张数
    num_cards_left = [_get_one_hot_array(infoset.num_cards_left[p], __HAND_CARD_NUM__) for p in __PLAY_ROLES__]
    num_cards_left = np.concatenate(num_cards_left, axis=0)
    
    #分数归一化
    bid_score = _get_one_hot_array(infoset.bid_score//5, 40)
    game_score = _get_one_hot_array(infoset.game_score//5, 40)
    # win_score_distance = _get_one_hot_array(max(0, (infoset.bid_score - infoset.game_score-5)//5), 40)
    # lose_score_distance = _get_one_hot_array(max(0, (infoset.game_score - infoset.bid_score+5)//5), 40)
    remain_score = _get_one_hot_array((200 - infoset.game_score)//5, 40)
    score_info = np.hstack((
                        bid_score, 
                        game_score, 
                        # win_score_distance, 
                        # lose_score_distance, 
                        remain_score))
    
    #游戏进度
    game_period = _get_one_hot_array(__HAND_CARD_NUM__ - max(infoset.num_cards_left[r] for r in __PLAY_ROLES__), 
                                     __HAND_CARD_NUM__)

    
    # 合法动作
    legal_actions = [[] for i in range(__WRONG__)]
    legal_types, legal_type_val = [], []
    for tp, actions in enumerate(infoset.legal_actions):
        if tp >= __WRONG__:
            continue
        for act in actions:            
            legal_actions[tp].append(cards2matrix(act))            
        if len(legal_actions[tp]) > 0: 
            legal_types.append(_get_one_hot_array(tp, __WRONG__))
            legal_type_val.append(tp)
    num_legal_actions = len(legal_type_val)
    legal_types_batch = np.array(legal_types)
    
    for i in range(len(legal_actions)):
        legal_actions[i] = np.array(legal_actions[i])
    
    #出牌序列
    action_seq_x, seats_seq_z = _process_action_seq(infoset.card_play_action_seq, length=10)
    
    
    x = np.vstack((
                    my_handcards,# 2*4*15 = 120
                    major_cards,
                    played_cards,
                    play_hand_mask_cards,
                    other_handcards,
                    public_cards,
                    remain_score_cards,
                    last_round_play_cards,
                    round_play_cards,
                    action_seq_x,
                  ))

    z = np.hstack((
                    my_seat,# 4
                    round_play_lead_seat,# 4
                    my_play_order,
                    game_period, #25
                    num_cards_left,# 25*4
                    score_info,# 40*2
                    seats_seq_z,# 10*4
                ))

    
    x_batch = np.repeat(
        x[np.newaxis, :, :, :],
        num_legal_actions, axis=0)
    
    z_batch = np.repeat(
        z[np.newaxis, :],
        num_legal_actions, axis=0)
    z_batch = np.concatenate((legal_types_batch, z_batch), axis=1)
    
    
    obs = {
        'position': infoset.player_position,
        'x': x.astype(np.int8),
        'z': z.astype(np.int8),
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': legal_actions,
        'legal_types': legal_type_val,
    }
    return obs

def _get_bid_obs_resnet(infoset):
    my_handcards = cards2matrix(infoset.player_hand_cards)
    
    
    my_seat_one_hot = _get_one_hot_array(infoset.seat + 1, __PLAYER_COUNT__)
    banker_seat_one_hot = _get_one_hot_array(infoset.banker_seat + 1, __PLAYER_COUNT__)
    
    #合法动作mask：#当前主花色 one-hot: [不叫, s, h, c, d, n] 
    _my_action_batch = []
    for action in infoset.legal_actions:
        _my_action_batch.append(cards2matrix(action[1]))
    _my_action_batch = np.array(_my_action_batch)
    num_legal_actions = len(infoset.legal_actions)
    
    bid_cards = []
    bid_seats = []
    for bid_act in infoset.bid_seq:
        bid_cards.append(cards2matrix(bid_act[1]))
        bid_seats.append(_get_one_hot_array(bid_act[0] + 1, __PLAYER_COUNT__))
    for i in range(len(infoset.bid_seq), 3):
        bid_cards.append(cards2matrix([]))
        bid_seats.append(_get_one_hot_array(0, __PLAYER_COUNT__))
    
    x = np.vstack((
                    my_handcards,# 2*4*15 = 120
                    np.vstack(bid_cards)
                      ))
    _x_batch = np.repeat(x[np.newaxis, :, :, :],
                        num_legal_actions, axis=0)
    
    z = np.hstack((
                    my_seat_one_hot,
                    banker_seat_one_hot,
                    np.hstack(bid_seats),
                ))
    z_batch = np.repeat(
        z[np.newaxis, :],
        num_legal_actions, axis=0)
    # my_action_batch = np.repeat(
    #         _my_action_batch[np.newaxis, :, :],
    #         num_legal_actions, axis=0)
    x_batch = np.concatenate((_my_action_batch, _x_batch), axis=1)
        
    obs = {
        'position': infoset.player_position,
        'x': x.astype(np.int8),
        'z': z.astype(np.int8),
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
    }
    return obs

def _get_cover_obs_resnet(infoset):
    bid_score = infoset.bid_score
    major = infoset.major
    level = infoset.level
    
    my_seat_one_hot = _get_one_hot_array(infoset.seat + 1, __PLAYER_COUNT__)
    
    hand_cards = cards2matrix(infoset.player_hand_cards, major=major, level=level)    
    legal_actions = cards2matrix(infoset.player_hand_cards, major=major, level=level)
    major_cards = cards2matrix(infoset.majorCards, major=major, level=level)
    
    bid_cards = []
    bid_seats = []
    for bid_act in infoset.bid_seq:
        bid_cards.append(cards2matrix(bid_act[1]))
        bid_seats.append(_get_one_hot_array(bid_act[0] + 1, __PLAYER_COUNT__))
    for i in range(len(infoset.bid_seq), 3):
        bid_cards.append(cards2matrix([]))
        bid_seats.append(_get_one_hot_array(0, __PLAYER_COUNT__))
        
        
    x = np.vstack((
                    hand_cards,# 2*4*15 = 120
                    major_cards,
                    np.vstack(bid_cards)
                    ))
    x_batch = np.vstack((
                    hand_cards,# 2*4*15 = 120
                    major_cards,
                    np.vstack(bid_cards)
                    ))
    x_batch = np.repeat(x[np.newaxis, :, :, :],
                        1, axis=0)
    
    # bid_score = _get_one_hot_array(bid_score//5, 40)
    z = np.hstack((
                    my_seat_one_hot,
                    np.hstack(bid_seats),
                ))
    z_batch = np.repeat(z[np.newaxis, :],
                        1, axis=0)
    
    
    obs = {
        'position': infoset.player_position,
        'x': x.astype(np.int8),
        'z': z.astype(np.int8),
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': legal_actions,
    }
    return obs
