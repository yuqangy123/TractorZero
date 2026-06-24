from collections import Counter, OrderedDict, deque
import numpy as np
import torch
from itertools import chain
from rlcard.games.tractors.env.utils import *
import logging, random
import traceback
from rlcard.envs import Env
from rlcard.games.tractors.env.botzone import tractorGame as tractors
import copy
import os, math
from  utils import *

os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5.0'#设置环境变量增加超时时间

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('tractors_env_belief')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

class BidInfoSet(object):
    def __init__(self, player_position):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # The hand cands of the current player. A list.
        self.player_hand_cards = None
        
        self.bid_score = 0
        
        self.mask_bid_score = None
        

class CoverInfoSet(object):
    def __init__(self, player_position):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # The hand cands of the current player. A list.
        self.player_hand_cards = None
        
        self.bid_score = 0
        

class InfoSet(object):
    def __init__(self, player_position):
        self.player_position = player_position
        
        self.player_hand_cards = None
        
        self.public_cards = None
        
        self.played_cards = None
        
        self.other_hand_cards = None

        self.remain_score_cards = None

        self.round_play_cards = None

        self.last_round_play_cards = None
        
        self.mask_cards = None
        
        self.play_rights_seat = None
        
        self.num_cards_left = None
        
        self.bid_score = None
        
        self.game_score = None
        
class GameEnv(tractors):
    ''' tractor Environment
    '''

    def __init__(self):
        self.num_wins = {'banker': 0,
                         'banker_up': 0,
                         'banker_down': 0,
                         }

        self.num_scores = {'banker': 0,
                         'banker_up': 0,
                         'banker_down': 0,
                         }
        
        self.bidding_player_position = 0
        self.reset()

    def reset(self):
        self.stage = 'bid'
        
        self.bid_over = False
        self.cover_over = False
        self.game_over = False

        self.coving_player_position = 0
        self.acting_player_position = 0
        
        self.bid_infoset = BidInfoSet('bid')
        self.cover_infoset = CoverInfoSet('cover')
        self.game_infoset = {'banker':InfoSet('banker'), 'banker_up':InfoSet('banker_up'), 'banker_down':InfoSet('banker_down')}
        
        self.position = ["banker", 'banker_up', 'banker_down', 'bid', 'cover']

        remain_score_cards = [16, 17, 18, 19, 36, 37, 38, 39, 48, 49, 50, 51]
        remain_score_cards.extend([c+54 for c in remain_score_cards])
        self.remain_score_cards = remain_score_cards

        self.round_play_cards = {'banker':[], 'banker_up':[], 'banker_down':[]}
        self.last_round_play_cards = {'banker':[], 'banker_up':[], 'banker_down':[]}
        
        self.mask_cards = {'banker':range(108), 'banker_up':range(108), 'banker_down':range(108)}
       
        super().reset()
        
        #初始化bid_infoset
        self.bid_infoset.player_hand_cards = self.getPlayerHandCards(self.getPlayerPosition())
        self.bid_infoset.bid_score = 80#最少叫80分
        self.bid_infoset.mask_bid_score = 80
        
    def getError(self):
        return ''


    def step(self, response):
        env = self
        
        response.insert(0, env.getPlayerPosition())
        
        #step前置处理，更新env缓存数据
        last_stage = env.getStage()
        if last_stage == 'play':
            play_cards = response[1]
            play_pos = env.getPlayerPosition()
            banker_pos = env.getBanker()
            rule = 'banker'
            if banker_pos != play_pos: rule = 'banker_up' if (play_pos+1)%__PLAYER_COUNT__ == banker_pos else 'banker_down'
            for c in play_cards:
                if c in self.remain_score_cards: self.remain_score_cards.remove(c)
            self.round_play_cards[rule] = play_cards[:]
            
        elif last_stage == 'cover':
            play_pos = env.getBanker()
            for i, r in enumerate(['banker', 'banker_down', 'banker_up']):                            
                hand_cards = env.getPlayerHandCards((play_pos+i)%__PLAYER_COUNT__)
                for c in hand_cards:self.mask_cards[r].remove(c)
            
        #step
        super().step(response)
        err = env.getError()
        if len(err)>0:
            print(err[len(err)-1])
            # env.reset()
            # env.step(response)
            raise Exception("An error occurred")

        
            
        #最新状态
        self.stage = env.getStage()
        
        #叫分阶段
        if self.stage == "bid":
            play_pos = env.getPlayerPosition()
            self.bidding_player_position = play_pos
            self.bid_infoset.player_position = play_pos
            self.bid_infoset.player_hand_cards = env.getPlayerHandCards(play_pos)
            self.bid_infoset.bid_score = env.getLeastBidScore()
            self.bid_infoset.mask_bid_score = self.bid_infoset.bid_score - 5#可叫分范围
            
            
        #埋牌阶段
        elif self.stage == "cover":
            self.bid_over = True
            play_pos = env.getPlayerPosition()
            self.cover_infoset.player_position = play_pos
            self.cover_infoset.player_hand_cards = env.getPlayerHandCards(play_pos)
            self.cover_infoset.bid_score = env.getLeastBidScore()
            self.coving_player_position = play_pos
                                
        
        elif self.stage == 'play':
            self.cover_over = True                        
            play_pos = env.getPlayerPosition()
            banker_pos = env.getBanker()
            rule = 'banker'
            if banker_pos != play_pos: rule = 'banker_up' if (play_pos+1)%__PLAYER_COUNT__ == banker_pos else 'banker_down'
            
            self.game_infoset[rule].player_position = rule
            self.game_infoset[rule].player_hand_cards = env.getPlayerHandCards(play_pos)
            self.game_infoset[rule].public_cards = env.getPublicCards()
            self.game_infoset[rule].played_cards = env.getPlayedCards(play_pos)
            self.game_infoset[rule].other_hand_cards = env.getPlayedCards((play_pos+1)%__MAX_SCORE__) + env.getPlayedCards((play_pos+2)%__MAX_SCORE__)
            self.game_infoset[rule].remain_score_cards = self.remain_score_cards[:]
            
            
            self.game_infoset[rule].round_play_cards = {}
            self.game_infoset[rule].last_round_play_cards = {}
            self.game_infoset[rule].mask_cards = {}
            self.game_infoset[rule].num_cards_left = {}
            
            #如果是垫牌，说明该玩家已经没有该花色            
            history_curr = env.getCurrRoundPlayHistory()
            if response[2] == __DISCARD__ and len(history_curr) > 1:
                majors = env.getMajorCards()
                if history_curr[0][0] in majors:
                    for card in majors:
                        if card in self.mask_cards[rule]:self.mask_cards[rule].remove(card)
                else:
                    play_suit = (history_curr[0][0])%4
                    for c in range(14):
                        for bt in [0, 54]:
                            card = c+play_suit+bt
                            if card in self.mask_cards[rule]:
                                self.mask_cards[rule].remove(card)
                        
            for i, r in enumerate(['banker', 'banker_down', 'banker_up']):
                self.game_infoset[rule].round_play_cards[r] =  self.round_play_cards[r][:]
                
                self.game_infoset[rule].last_round_play_cards[r] =  self.last_round_play_cards[r][:]

                for c in response[1]:
                    if c in self.mask_cards[r]:
                        self.mask_cards[r].remove(c)
                self.game_infoset[rule].mask_cards[r] =  self.mask_cards[r][:]

                self.game_infoset[rule].num_cards_left[r] = env.getPlayerLeftHandCards((banker_pos+1)%__PLAYER_COUNT__)
                
            self.game_infoset[rule].play_rights_seat = env.getFristPlaySeat()
            
            self.game_infoset[rule].bid_score = env.getLeastBidScore()
            
            self.game_infoset[rule].game_score = env.getGameScore()
            
            
            hold = env.getPlayerHandCards(play_pos)
            playedCards = env.getLegalPlayCard(history_curr, hold, env.getLevel())
            self.game_infoset[rule].legal_actions = playedCards
            
            self.acting_player_position = play_pos
                        
        
        #一回合结束
        elif self.stage == 'roundend':
            for r, cards in self.round_play_cards.items():
                self.last_round_play_cards[r] = cards[:]
                self.round_play_cards[r] = []
                
        elif self.stage == 'gameend':
            self.game_over = True
            self.bidding_player_position = env.getPlayerPosition()