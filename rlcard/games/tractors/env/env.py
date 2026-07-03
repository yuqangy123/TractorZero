from collections import Counter
import numpy as np

from .game import GameEnv
from .utils import *



deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])


class Env:

    def __init__(self, objective):
        self.objective = objective

        # Initialize the internal environment
        self._env = GameEnv()
        self.total_round = 0
        self.infoset = None

    def reset(self, model, device, flags=None):
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
            #         "banker": self._get_reward("banker"),
            #         "banker_down": self._get_reward("banker_down"),
            #         "banker_up": self._get_reward("banker_up"),
            #         "bid": self._get_reward_bid(),
            #         "cover": self._get_reward_cover(),
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
        banker = self._env.getBanker()
        bid_score = self._env.getLeastBidScore()
        round_score = self._env.getLastRoundScore()
        game_score = self._env.getGameScore()
        play_rights = self._env.getFristPlaySeat()

        r = round_score/100.0#4*K + 4*10 + 4*5
        if 0.0 < round_score:
            if game_score <= bid_score:
                pro_score = (game_score/bid_score) ** 1.3
            else:
                pro_score = (game_score/200.) ** 0.7
                
            r += pro_score*0.5
        else:
            r = -self._env.getPlayerLeftHandCards(banker)/__HAND_CARD_NUM__ * 0.5
            
        mult = -1 if banker == play_rights else 1
        
        reward = {}
        reward['banker'].append(r * mult)
        reward['banker_down'].append(r * mult)
        reward['banker_up'].append(r * mult)
        
        return reward

    def _get_reward(self):
        banker = self._env.getBanker()
        bid_score = self._env.getLeastBidScore()#叫分
        game_score = self._env.getGameScore()#局内得分
        
        banker_win = bid_score > game_score
        
        public_score = 0
        #[16, 17, 18, 19, 36, 37, 38, 39, 48, 49, 50, 51]
        for c in self._env.getPublicCards():
            if c in [16, 17, 18, 19, 16+54, 17+54, 18+54, 19+54]:public_score += 5
            elif c in [36, 37, 38, 39, 36+54, 37+54, 38+54, 39+54]:public_score += 10
            elif c in [48, 49, 50, 51, 48+54, 49+54, 50+54, 51+54]:public_score += 10
        
        mult = 1. if banker_win else -1.
        
        reward = {}
        #bid和cover都是以banker为actor
        reward['cover_public_score'] = public_score/100. * mult
        reward['cover'] = 1.0 if True == banker_win else -1.0
        
        if bid_score == game_score:
            reward['bid'] = 5./bid_score
        else:
            bid_diff_ratio = (bid_score - game_score) / bid_score
            reward['bid'] = np.clip(bid_diff_ratio, -1.0, 1.0)
        
        if banker_win:
            end_score = self._env.getEndingScore(banker)
            reward['banker'] = end_score
            reward['banker_down'] = -end_score
            reward['banker_up'] = -end_score
        else:
            end_score = self._env.getEndingScore((banker+1)%__PLAYER_COUNT__)
            reward['banker'] = -end_score
            reward['banker_down'] = end_score
            reward['banker_up'] = end_score
        
        return reward
        
    @property
    def _game_infoset(self):
        return self._env.game_infoset

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
    if infoset.player_position not in ["banker", 'banker_up', 'banker_down', 'bid', 'cover']:
        raise ValueError('')
    if stage == 'bid':
        return _get_bid_obs_resnet(infoset)
    elif stage == 'cover':
        return _get_cover_obs_resnet(infoset)
    else:
        if infoset.player_position == 'banker':
            return _get_banker_obs_resnet(infoset)
        else:
            return _get_idler_obs_resnet(infoset)
        
    
def _get_one_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards)
    if num_left_cards > 0:
        one_hot[num_left_cards - 1] = 1

    return one_hot

def _get_one_hot_array_ex(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards)
    if num_left_cards > 0:
        one_hot[:num_left_cards] = 1

    return one_hot

def _get_custom_one_hot_array_ex(num_left_cards, max_num_cards, one_value):
    one_hot = np.zeros(max_num_cards)
    one_hot[:] = one_value
    if num_left_cards > 0:
        one_hot[:num_left_cards] = 1

    return one_hot

def _get_full_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards)
    if num_left_cards > 0:
        one_hot[:num_left_cards] = 1

    return one_hot

def _get_banker_obs_resnet(infoset):
    # num_legal_actions = len(infoset.legal_actions)
    # my_handcards_batch = np.repeat(my_handcards[np.newaxis, :,:,:],
    #                                num_legal_actions, axis=0)
    major = infoset.major
    level = infoset.level
    my_handcards = cards2matrix(infoset.player_hand_cards, level=level, major=major)
    
    banker_played_cards = cards2matrix(infoset.played_cards['banker'], level=level, major=major)
    banker_up_played_cards = cards2matrix(infoset.played_cards['banker_up'], level=level, major=major)
    banker_down_played_cards = cards2matrix(infoset.played_cards['banker_down'], level=level, major=major)
    
    other_handcards = cards2matrix(infoset.other_hand_cards, level=level, major=major)
    
    public_cards = cards2matrix(infoset.public_cards, level=level, major=major)
    
    remain_score_cards = cards2matrix(infoset.remain_score_cards, level=level, major=major)
    
    banker_up_round_play_cards = cards2matrix(infoset.round_play_cards['banker_up'], level=level, major=major)
    banker_down_round_play_cards = cards2matrix(infoset.round_play_cards['banker_down'], level=level, major=major)
    
    banker_up_last_round_played_cards = cards2matrix(infoset.last_round_played_cards['banker_up'], level=level, major=major)
    banker_down_last_round_played_cards = cards2matrix(infoset.last_round_played_cards['banker_down'], level=level, major=major)
    
    banker_up_mask_cards = infoset.mask_cards['banker_up']
    banker_down_mask_cards = infoset.mask_cards['banker_down']
    
    play_rights = _get_one_hot_array(infoset.play_rights_seat, 3)
    
    
    legal_actions = [[] for _ in range(__WRONG__)]
    for k, actions in infoset.legal_actions.items():
        for act in actions:
            legal_actions[k].append(cards2matrix(act), level=level, major=major)
    legal_actions = np.array(legal_actions)
        
    
    #剩余牌张数
    banker_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left['banker'], 28)

    banker_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left['banker_up'], 28)

    banker_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left['banker_down'], 28)
    num_cards_left = np.hstack((
                         banker_num_cards_left,
                         banker_up_num_cards_left,
                         banker_down_num_cards_left))

    #分数归一化
    bid_score = _get_one_hot_array(infoset.bid_score//5, 40)
    get_score = _get_one_hot_array(infoset.get_score//5, 40)
    win_score_distance = _get_one_hot_array(max(0, (infoset.bid_score - infoset.get_score-5)//5), 40)
    lose_score_distance = _get_one_hot_array(max(0, (infoset.get_score - infoset.bid_score+5)//5), 40)
    remain_score = _get_one_hot_array((200 - infoset.get_score)//200, 40)
    score_left = np.hstack((
                        bid_score, 
                        get_score, 
                        win_score_distance, 
                        lose_score_distance, 
                        remain_score))
    
    #游戏进度
    game_period = (28.0 - max(infoset.num_cards_left['banker'], infoset.num_cards_left['banker_up'], infoset.num_cards_left['banker_down']))/28.0

    x_no_action = np.vstack((
                    my_handcards,# 2*4*15 = 120
                    banker_played_cards,
                    banker_up_played_cards,
                    banker_down_played_cards,
                    other_handcards,
                    public_cards,
                    remain_score_cards,
                    banker_up_round_play_cards,
                    banker_down_round_play_cards,
                    banker_up_last_round_played_cards,
                    banker_down_last_round_played_cards,
                    banker_up_mask_cards,
                    banker_down_mask_cards,
                  ))

    z = np.hstack((
                    play_rights,# 3
                    game_period, #1
                    num_cards_left,# 28*3 = 84
                    score_left,# 40*5 = 200                    
                ))

    # _z_batch = np.repeat(
    #     z[np.newaxis, :, :],
    #     num_legal_actions, axis=0)
    # my_action_batch = my_action_batch[:, np.newaxis, :]
    # z_batch = np.concatenate((my_action_batch, _z_batch), axis=1)
    obs = {
        'position': infoset.player_position,
        'x': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
        'legal_actions': legal_actions,
    }
    return obs

def _get_idler_obs_resnet(infoset):
    major = infoset.major
    level = infoset.level
    
    partner_position = 'banker_up' if infoset.player_position == 'banker_down' else 'banker_down'
    my_handcards = cards2matrix(infoset.player_hand_cards, level=level, major=major)
    
    banker_played_cards = cards2matrix(infoset.played_cards['banker'], level=level, major=major)
    banker_up_played_cards = cards2matrix(infoset.played_cards['banker_up'], level=level, major=major)
    banker_down_played_cards = cards2matrix(infoset.played_cards['banker_down'], level=level, major=major)
    
    other_handcards = cards2matrix(infoset.other_hand_cards, level=level, major=major)
        
    remain_score_cards = cards2matrix(infoset.remain_score_cards, level=level, major=major)
    
    banker_round_play_cards = cards2matrix(infoset.round_play_cards['banker'], level=level, major=major)
    partner_round_play_cards = cards2matrix(infoset.round_play_cards[partner_position], level=level, major=major)
    
    banker_last_round_played_cards = cards2matrix(infoset.last_round_played_cards['banker'], level=level, major=major)
    partner_last_round_played_cards = cards2matrix(infoset.last_round_played_cards[partner_position], level=level, major=major)
    
    banker_mask_cards = infoset.mask_cards['banker']
    partner_mask_cards = infoset.mask_cards[partner_position]
    
    play_rights = _get_one_hot_array(infoset.play_rights_seat, 3)
        
    legal_actions = []
    for j, action in enumerate(infoset.legal_actions):
        legal_actions.append(cards2matrix(action), level=level, major=major)
    
    #剩余牌张数
    banker_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left['banker'], 28)

    banker_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left['banker_up'], 28)

    banker_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left['banker_down'], 28)
    num_cards_left = np.hstack((
                         banker_num_cards_left,
                         banker_up_num_cards_left,
                         banker_down_num_cards_left))

    #分数归一化
    bid_score = _get_one_hot_array(infoset.bid_score//5, 40)
    get_score = _get_one_hot_array(infoset.get_score//5, 40)
    lose_score_distance = _get_one_hot_array(max(0, (infoset.bid_score - infoset.get_score)//5), 40)
    win_score_distance = _get_one_hot_array(max(0, (infoset.get_score - infoset.bid_score+5)//5), 40)
    remain_score = _get_one_hot_array((200 - infoset.get_score)//200, 40)
    score_left = np.hstack((
                        bid_score, 
                        get_score, 
                        win_score_distance, 
                        lose_score_distance, 
                        remain_score))
    
    #游戏进度
    game_period = (28.0 - max(infoset.num_cards_left['banker'], infoset.num_cards_left['banker_up'], infoset.num_cards_left['banker_down']))/28.0

    x_no_action = np.vstack((
                    my_handcards,# 2*4*15 = 120
                    banker_played_cards,
                    banker_up_played_cards,
                    banker_down_played_cards,
                    other_handcards,
                    remain_score_cards,
                    banker_round_play_cards,
                    partner_round_play_cards,
                    banker_last_round_played_cards,
                    partner_last_round_played_cards,
                    banker_mask_cards,
                    partner_mask_cards,
                  ))

    z = np.hstack((
                    play_rights,# 3
                    game_period, #1
                    num_cards_left,# 28*3 = 84
                    score_left,# 40*5 = 200                    
                ))

    # _z_batch = np.repeat(
    #     z[np.newaxis, :, :],
    #     num_legal_actions, axis=0)
    # my_action_batch = my_action_batch[:, np.newaxis, :]
    # z_batch = np.concatenate((my_action_batch, _z_batch), axis=1)
    obs = {
        'position': infoset.player_position,
        'x': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
        'legal_actions': legal_actions,
    }
    return obs

def _get_bid_obs_resnet(infoset):
    my_handcards = cards2matrix(infoset.player_hand_cards)
    last_bid_score = _get_one_hot_array_ex(infoset.bid_score//5 + 1, 40 + 1)#1是在最开头加一个不叫
    mask_bid_score = _get_one_hot_array_ex(infoset.mask_bid_score//5 + 1, 40 + 1)
    #0号位表示不叫
    
    #合法动作mask
    legal_actions = np.zeros(40 + 1)
    legal_actions[infoset.mask_bid_score//5 + 1:] = -999.
    legal_actions = np.expand_dims(legal_actions, axis=0)
    
    x_no_action = np.vstack((
                    my_handcards,# 2*4*15 = 120
                  ))
    x_no_action = np.expand_dims(x_no_action, axis=0)
    
    
    z = np.hstack((
                    last_bid_score,
                    mask_bid_score,
                ))
    z = np.expand_dims(z, axis=0)

    obs = {
        'position': infoset.player_position,
        'x': x_no_action.astype(np.float32),
        'z': z.astype(np.float32),
        'legal_actions': legal_actions,
    }
    return obs

def _get_cover_obs_resnet(infoset):
    bid_score = infoset.bid_score
    my_handcards = cards2matrix(infoset.player_hand_cards)
    
    legal_actions = cards2matrix(infoset.player_hand_cards)
    
    #分数归一化
    bid_score = _get_one_hot_array(bid_score//5, 40)
    
    x_no_action = np.vstack((
                    my_handcards,# 2*4*15 = 120
                  ))

    z = np.hstack((
                    bid_score                
                ))

    obs = {
        'position': infoset.player_position,
        'x': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
        'legal_actions': legal_actions,
    }
    return obs