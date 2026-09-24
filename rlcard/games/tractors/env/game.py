from rlcard.games.tractors.env.utils import *
import logging
from rlcard.games.tractors.env.botzone import tractorGame as tractors
import os
from rlcard.games.tractors.env.utils import *

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
        
        self.bid_score = None
        
        self.major = None
        
        self.level = None
        
        self.majorCards = None
        
        self.seat = None
        
        self.banker_seat = None
        
        self.legal_actions = None
        
        self.bid_seq = None
        

class CoverInfoSet(object):
    def __init__(self, player_position):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # The hand cands of the current player. A list.
        self.player_hand_cards = None
        self.major = None
        self.level = None
        self.bid_score = None
        self.bid_seq = None
        self.majorCards = None
        

class InfoSet(object):
    def __init__(self, player_position):
        self.player_position = player_position
        self.seat = None
        self.banker_seat = None
        self.play_rule = None
        self.player_hand_cards = None        
        self.public_cards = None        
        self.played_cards = None        
        self.other_hand_cards = None
        self.remain_score_cards = None
        self.round_play_cards = None
        self.last_round_play_cards = None        
        self.mask_cards = None        
        self.round_play_lead_seat = None        
        self.num_cards_left = None        
        self.bid_score = None        
        self.game_score = None
        self.majorCards = None
        self.legal_actions = None
        self.card_play_action_seq = None
        self.play_order = None
        
class GameEnv(tractors):
    ''' tractor Environment
    '''

    

    def _role_of(self, play_pos, banker_pos):
        if play_pos == banker_pos:
            return 'banker'
        if (play_pos + 2) % __PLAYER_COUNT__ == banker_pos:
            return 'banker_op'
        if (play_pos + 1) % __PLAYER_COUNT__ == banker_pos:
            return 'banker_up'
        return 'banker_down'

    def _role_of_seat(self, play_pos, banker_pos):
            if play_pos == banker_pos:
                return 0
            if (play_pos + 2) % __PLAYER_COUNT__ == banker_pos:
                return 2
            if (play_pos + 1) % __PLAYER_COUNT__ == banker_pos:
                return 3
            return 1
        
    def _rule_of_position(self, rule):
        if rule in __PLAY_ROLES__:
            return (self.getBanker() + __PLAY_ROLES__.index(rule)) % __PLAYER_COUNT__
        
    
    def __init__(self):
        super().__init__()
        
        # self.reset()

    def reset(self):
        super().reset()
                
        self.stage = 'bid'
        
        self.bid_over = False
        self.cover_over = False
        self.game_over = False

        self.player_rule = 'banker'
        
        self.bid_infoset = BidInfoSet('bid')
        self.cover_infoset = CoverInfoSet('cover')        
        self.game_infoset = {role:InfoSet(role) for role in __PLAY_ROLES__}

        self.remain_score_cards = __SCORE__CARD__[:]

        self.round_play_cards = {role:[] for role in __PLAY_ROLES__}
        self.last_round_play_cards = {role:[] for role in __PLAY_ROLES__}
        
        #有效牌mask: mask_cards[视角角色][目标角色] = 该视角下目标角色可能持有的牌
        self.mask_cards = {role:{player_role:set(range(108)) for player_role in __PLAY_ROLES__} for role in __PLAY_ROLES__}
        
        #初始化bid_infoset
        self.bid_infoset.player_hand_cards = self.getPlayerHandCards(self.getPlayerPosition())
        self.bid_infoset.banker_seat = self.getBanker()
        self.bid_infoset.seat = self.getPlayerPosition()
        self.bid_infoset.major = self.getMajorColor()
        self.bid_infoset.level = self.getLevel()
        self.bid_infoset.legal_actions = []
        bidActions = self.getLegalBidActions(self.getPlayerPosition())
        for act in bidActions:
            self.bid_infoset.legal_actions.append([act, self.getMajorCards(act, self.bid_infoset.level)])
        self.bid_infoset.bid_seq = []
        for bid_act in self.getBidSeq():
            self.bid_infoset.bid_seq.append([bid_act[0], self.getMajorCards(bid_act[1], self.bid_infoset.level)])
        self.bid_infoset.bid_score = 80#闲家升级分数线固定80分
        
        
        
    def getError(self):
        # 汇总底层记录的报错（每条为 [reason, endingScores]），供 step 后判非法
        return [err for errs in self.errored for err in errs]
    
    #step前置处理，更新env缓存数据
    def step_front(self, response):
        env = self
        
        last_stage = env.getStage()        
        if last_stage == 'bid':
            response.insert(0, env.getPlayerPosition())
            
        elif last_stage == 'cover':
            banker_pos = env.getBanker()            
            #mask掉玩家自身视角下的我的手牌(25+8)：自己那一格即自己的手牌，其余角色剔除自己已知的牌
            for i in range(__PLAYER_COUNT__):
                my_r = self._role_of(i, banker_pos)
                hand_cards = set(env.getPlayerHandCards(i))
                self.mask_cards[my_r][my_r] = hand_cards.copy()
                for play_r in __PLAY_ROLES__:
                    if play_r != my_r:
                        self.mask_cards[my_r][play_r] -= hand_cards
            
                
            response.insert(0, banker_pos)
        
        elif last_stage == 'play':
            banker_pos = env.getBanker()
            #mask掉已打出的牌：出牌是公开信息，对所有视角的候选牌做幂等剔除
            played = set(response[0])
            for i in range(__PLAYER_COUNT__):
                my_r = self._role_of(i, banker_pos)
                for player_r in __PLAY_ROLES__:
                    self.mask_cards[my_r][player_r] -= played

            response.insert(0, env.getPlayerPosition())
    
    def step(self, response):
        env = self
        self.step_front(response)
            
        #step
        super().step(response)
        err = env.getError()
        if len(err)>0:
            print(err[len(err)-1])
            # env.reset()
            # env.step(response)
            raise Exception(f"An step error occurred!\n{err}")

        
            
        #最新状态
        stage = env.getStage()
        self.stage = stage
        
        #叫主阶段
        if stage == "bid":
            play_pos = env.getPlayerPosition()
            banker_pos = env.getBanker()
            self.bid_infoset.seat =  self._role_of_seat(play_pos, banker_pos)
            self.bid_infoset.banker_seat = 0
            self.bid_infoset.player_hand_cards = env.getPlayerHandCards(play_pos)
            
            self.bid_infoset.level = env.getLevel()
            self.bid_infoset.major = env.getMajorColor()
            self.bid_infoset.majorCards = env.getMajorCards(env.getMajorColor(), env.getLevel())
            
            self.bid_infoset.legal_actions = []
            bidActions = env.getLegalBidActions(play_pos)
            for act in bidActions:
                self.bid_infoset.legal_actions.append([act, env.getMajorCards(act, self.bid_infoset.level)])
             
            self.bid_infoset.bid_seq = []
            for bid_act in env.getBidSeq():
                self.bid_infoset.bid_seq.append([self._role_of_seat(bid_act[0], banker_pos), env.getMajorCards(bid_act[1], self.bid_infoset.level)])
                        
        #埋牌阶段
        elif stage == "cover":
            self.bid_over = True
            banker_pos = 0
            play_pos = env.getBanker()
            self.cover_infoset.seat =  0
            self.cover_infoset.banker_seat = 0
            self.cover_infoset.player_hand_cards = env.getPlayerHandCards(play_pos)
            self.cover_infoset.bid_score = self.bid_infoset.bid_score
            self.cover_infoset.major = env.getMajorColor()
            self.cover_infoset.level = env.getLevel()
            self.cover_infoset.majorCards = env.getMajorCards()
            
            self.cover_infoset.bid_seq = []
            for bid_act in env.getBidSeq():
                self.cover_infoset.bid_seq.append([self._role_of_seat(bid_act[0], banker_pos), env.getMajorCards(bid_act[1], self.bid_infoset.level)])
            
        #出牌准备阶段
        elif stage == "ready" or stage == 'play':
            self.cover_over = True
            
            history_curr = env.getCurrRoundPlayHistory()
            if len(history_curr) > 0:
                rights_eat = env.getFristPlaySeat()
                last_play_cards = history_curr[-1]
                
                #更新剩余分数牌
                for c in last_play_cards:
                    if c in self.remain_score_cards: self.remain_score_cards.remove(c)
                
                rule = self._role_of((rights_eat + len(history_curr) - 1)%__PLAYER_COUNT__, env.getBanker())
                self.round_play_cards[rule] = last_play_cards[:]# 如果是甩牌 则按实际出的牌算
            
            play_pos = env.getPlayerPosition()
            banker_pos = env.getBanker()
            level = env.getLevel()
            major = env.getMajorColor()
            rule = self._role_of(play_pos, banker_pos)
            self.game_infoset[rule].seat =  self._role_of_seat(play_pos, banker_pos)
            self.game_infoset[rule].banker_seat = 0
            self.game_infoset[rule].play_order = len(history_curr)
            
            self.game_infoset[rule].major = major
            self.game_infoset[rule].level = level
            
            self.game_infoset[rule].play_rule = rule
            self.game_infoset[rule].public_cards = env.getPublicCards() if rule == 'banker' else []
            self.game_infoset[rule].player_hand_cards = env.getPlayerHandCards(play_pos)
            
            self.game_infoset[rule].other_hand_cards = []
            for i in range(1, __PLAYER_COUNT__):
                self.game_infoset[rule].other_hand_cards.extend(env.getPlayerHandCards((play_pos+i)%__PLAYER_COUNT__))
            self.game_infoset[rule].remain_score_cards = self.remain_score_cards[:]
                        
            self.game_infoset[rule].round_play_cards = {}
            self.game_infoset[rule].last_round_play_cards = {}
            self.game_infoset[rule].mask_cards = {}
            self.game_infoset[rule].num_cards_left = {}
            self.game_infoset[rule].played_cards = {}
            
            #如果是垫牌，说明该玩家已经没有该花色
            if len(response) > 2 and response[2] == __DISCARD__ and len(history_curr) > 0:
                # 推断对象是刚垫牌的人(response[0])，而不是下一个要出牌的人
                discarder_r = self._role_of(response[0], banker_pos)
                lead_card = history_curr[0][0]
                majors = env.getMajorCards()
                if lead_card in majors:
                    unknown = set(majors)
                else:
                    # 花色取 (牌号%54)%4，第二副牌牌号是+54，不能直接用%4
                    play_suit = (lead_card % 54) % 4
                    # 级牌属于主牌，不参与该花色推断
                    level_rank = __CARDSCALE__.index(level)
                    unknown = {r*4 + play_suit for r in range(13) if r != level_rank} | \
                              {r*4 + play_suit + 54 for r in range(13) if r != level_rank}
                for viewer_r in __PLAY_ROLES__:
                    self.mask_cards[viewer_r][discarder_r] -= unknown
                                
            for i, r in enumerate(__PLAY_ROLES__):
                self.game_infoset[rule].round_play_cards[r] =  self.round_play_cards[r][:]
                self.game_infoset[rule].last_round_play_cards[r] =  self.last_round_play_cards[r][:]                
                self.game_infoset[rule].num_cards_left[r] = env.getPlayerLeftHandCards((banker_pos+i)%__PLAYER_COUNT__)
                self.game_infoset[rule].played_cards[r] = env.getPlayedCards((banker_pos+i)%__PLAYER_COUNT__)
                self.game_infoset[rule].majorCards = env.getMajorCards()
                self.game_infoset[rule].mask_cards[r] =  list(self.mask_cards[rule][r])
                
                
                
            self.game_infoset[rule].round_play_lead_seat = self._role_of_seat(env.getFristPlaySeat(), banker_pos)
            self.game_infoset[rule].bid_score = env.getLeastBidScore()
            self.game_infoset[rule].game_score = env.getGameScore()
            
            hold = env.getPlayerHandCards(play_pos)
            playedCards = env.getLegalPlayCard(history_curr, hold, env.getLevel())
            self.game_infoset[rule].legal_actions = playedCards
            
            
            play_history = env.getPlayHistory()
            self.game_infoset[rule].card_play_action_seq = [
                [self._role_of_seat(seat, banker_pos), cards, typ]
                for seat, cards, typ in play_history
            ]
            
            self.player_rule = rule
        
        #一回合结束
        elif stage == 'roundend':
            round_play_cards = env.getLastRoundPlayHistory()[-1]
        
            for c in round_play_cards:
                if c in self.remain_score_cards: self.remain_score_cards.remove(c)
                
            rule = self._role_of((env.getLastRoundPlaySeat()+__PLAYER_COUNT__-1)%__PLAYER_COUNT__, env.getBanker())
            self.round_play_cards[rule] = round_play_cards
            
            for r in __PLAY_ROLES__:
                self.last_round_play_cards[r] = self.round_play_cards[r][:]
                self.round_play_cards[r] = []
            
            
        elif stage == 'gameend' or stage == 'finalend':
            self.game_over = True
            
    def calc_hand_power(self, hand_cards):
        """
        计算一手牌的牌力(归一化到0~1，即平均每张牌的强度)。

        基础分:
        - 大王57、小王56;
        - 主牌: 40 + 在Major中的位置(Major按强度从小到大排列，主级牌高于其余级牌);
        - 副牌: 按点数2~A计2~14分(副牌最大强度14 < 主牌最低40，保证主牌价值显著更高)。

        结构加成: 用 getLegalPlayCard(history为空, 即首发)枚举本手的全部合法出牌牌型:
        - 拖拉机: 8(对子) + 14(基础) + 2*每多一对，4张拖拉机+22/张，越长牌力越大;
        - 对子 +8/张; 甩牌 +6/张(同一张牌只取最高牌型)。
        单张上限 51 + 8 + 14 + 2*(12-2) = 93，归一化时分母取93*手牌数，保证结果<=1
        (副牌每花色去掉级牌最多12对，故拖拉机最多12对)。
        """
        if len(hand_cards) == 0:
            return 0.0
        major = self.Major
        level = self.getLevel()

        pair_bonus = 8.0#对子加成
        tractor_bonus = 14.0#拖拉机基础加成
        extra_per_pair = 2.0#拖拉机每多一对的额外加成
        suspect_bonus = 6.0#甩牌加成

        #按首发枚举本手全部合法出牌牌型: 单张/对子/拖拉机/甩牌
        all_hands = self.getLegalPlayCard([], hand_cards, level)

        tractor_bonus_of = {}#牌号(n) -> 所属拖拉机的每张加成(长度越长加成越大)
        for play in all_hands[__TRACTOR__]:
            pairs = len(play) // 2
            bonus = pair_bonus + tractor_bonus + extra_per_pair * (pairs - 2)
            for c in play:
                n = c % 54
                if tractor_bonus_of.get(n, 0.0) < bonus:
                    tractor_bonus_of[n] = bonus
        pair_ns = set()
        for play in all_hands[__PAIR__]:
            pair_ns.update(c % 54 for c in play)
        suspect_ns = set()
        for play in all_hands[__SUSPECT__]:
            suspect_ns.update(c % 54 for c in play)

        power = 0.0
        for c in hand_cards:
            n = c % 54
            if n == 53:#大王
                base = 57.0
            elif n == 52:#小王
                base = 56.0
            else:
                poker = self.num2Poker(c)
                if poker in major:#主牌
                    base = 40.0 + major.index(poker)
                else:#副牌: 按点数2~A计2~14
                    base = 2.0 + self.pointorder.index(poker[1])
            if n in tractor_bonus_of:#拖拉机中的牌(加成随长度增长)
                power += base + tractor_bonus_of[n]
            elif n in pair_ns:#对子中的牌
                power += base + pair_bonus
            elif n in suspect_ns:#甩牌中的牌
                power += base + suspect_bonus
            else:#单张/散牌
                power += base
        return power / (93.0 * len(hand_cards))

    def encode_state_feature(self, rule):
        """
        设计一个"可比较"的状态向量，用于计算 delta。
        关键：状态向量要能反映局面变化。
        """
        env = self
        pos = self._rule_of_position(rule)
        hand_cards = env.getPlayerHandCards(pos)
        num_hand_cards_left = len(hand_cards)
        num_major_cards_left = env.getPlayerLeftMajorCards(pos)
        banker_pos = env.getBanker()
        remain_score = 0.
        for c in hand_cards:
            if c in __SCORE__CARD__[:8]:remain_score += 5.
            elif c in __SCORE__CARD__[8:]:remain_score += 10.
        
        remain_other_score = 0.
        for c in self.remain_score_cards:
            if c in __SCORE__CARD__[:8]:remain_other_score += 5.
            elif c in __SCORE__CARD__[8:]:remain_other_score += 10.

        # game_score = env.getGameScore(pos)
        
        lead_seat = env.getFristPlaySeat()
        my_lead = lead_seat == pos
        teammate_lead = lead_seat == (pos+2)%__PLAYER_COUNT__
        opponent_lead = not my_lead and not teammate_lead
        
        #计算牌力
        hand_power = self.calc_hand_power(hand_cards)
        
        return [
            num_hand_cards_left / __HAND_CARD_NUM__,           # 手牌数（归一化）
            num_major_cards_left / __HAND_CARD_NUM__,          # 主牌数（归一化）
            remain_score / 200.0,  # 手中分牌数
            remain_other_score / 200.0,  # 剩余未出分牌数
            # game_score / 200.0,  # 当前得分            
            my_lead,#我控牌
            teammate_lead,#队友控牌
            opponent_lead,#对手控牌
            hand_power,#手牌平均牌力(0~1)            
        ]