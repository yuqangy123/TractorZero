from collections import Counter, OrderedDict, deque
import numpy as np
import torch
from itertools import chain
from rlcard.games.tractors.utils import *
import logging, random
import traceback
from rlcard.envs import Env
from rlcard.games.tractors.tractor_botzone import tractorGame as tractors
import copy
import os, math
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

class TractorsEnv(tractors):
    ''' tractor Environment
    '''

    def __init__(self, config):
        # from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION
        # from rlcard.games.doudizhu.utils import cards2str, cards2str_with_suit
        
        # self._cards2str = cards2str
        # self._cards2str_with_suit = cards2str_with_suit
        # self._ACTION_2_ID = ACTION_2_ID
        # self._ID_2_ACTION = ID_2_ACTION
        super().__init__()
        # self.game = tractors()
        # self.state_shape = [[790], [901], [901]]
        # self.action_shape = [[54] for _ in range(self.num_players)]

    # def step(self, resp):
    #     super().step(resp)
    #     self.game.step(resp)
    #     # s = self.getStage()
    #     # if s == 'deal':
            
    # def getStage(self):
    #     return self.game.getStage()

    def reset(self):
        pass

    def getError(self):
        return ''


# infoset['hand_cards'] = hand_cards[banker]
# infoset['major_cards'] = major_card_mtx
# infoset['public_cards'] = public_cards
# infoset['played_score_cards'] = played_score_cards
# infoset['remain_score_cards'] = remain_score_cards
# infoset['history_play_cards'] = history_play_cards
# infoset['round_play_cards'] = round_play_cards
# infoset['last_play_cards'] = last_play_cards
# infoset['played_cards'] = played_cards
# infoset['mask_cards'] = mask_cards
# infoset['my_seat'] = get_one_hot_array(banker, __PLAYER_COUNT__)
# infoset['play_rights_seat'] = get_one_hot_array(play_rights_seat, __MAX_SCORE__)
# infoset['banker_seat'] = get_one_hot_array(banker, __PLAYER_COUNT__)
# infoset['score'] = get_full_hot_array(0, __MAX_SCORE__)
# infoset['win_score'] = get_full_hot_array(bid_score//5, __MAX_SCORE__)
def get_obs(infoset):
    num_legal_actions = len(infoset.legal_actions)
    
    my_handcards = infoset.hand_cards
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    major_cards = infoset.major_cards
    major_cards_batch = np.repeat(major_cards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    public_cards = infoset.public_cards
    public_cards_batch = np.repeat(public_cards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    played_score_cards = infoset.played_score_cards
    played_score_cards_batch = np.repeat(played_score_cards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    
    remain_score_cards = infoset.remain_score_cards
    remain_score_cards_batch = np.repeat(remain_score_cards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    round_play_cards = infoset.round_play_cards
    round_play_cards_batch = np.repeat(round_play_cards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    last_play_cards = infoset.last_play_cards
    last_play_cards_batch = np.repeat(last_play_cards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    played_cards = infoset.played_cards
    played_cards_batch = np.repeat(played_cards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    mask_cards = infoset.mask_cards
    mask_cards_batch = np.repeat(mask_cards[np.newaxis, :],
                                   num_legal_actions, axis=0)
    
    my_position_info = infoset.my_seat
    my_position_info_batch = np.repeat(my_position_info[np.newaxis, :],
                                    num_legal_actions, axis=0)
    
    banker_position_info = infoset.banker_seat
    banker_position_info_batch = np.repeat(banker_position_info[np.newaxis, :],
                                    num_legal_actions, axis=0)
    
    play_rights_seat_info = infoset.play_rights_seat
    play_rights_seat_info_batch = np.repeat(play_rights_seat_info[np.newaxis, :],
                                    num_legal_actions, axis=0)
    
    
    score = infoset.score
    score_batch = np.repeat(score[np.newaxis, :],
                                    num_legal_actions, axis=0)
    
    win_score = infoset.win_score
    win_score_batch = np.repeat(win_score[np.newaxis, :],
                                    num_legal_actions, axis=0)
    
    
    #闲家庄家
    banker = get_one_hot_array(1 if infoset.banker_seat == infoset.my_seat else 2, 2)
    banker_batch = np.repeat(banker[np.newaxis, :],
                                    num_legal_actions, axis=0)
    
    

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = cards2matrix(action)


    num_cards_left = np.hstack((
                         banker_num_cards_left,  # 20
                         player1_num_cards_left,  # 17
                         player2_num_cards_left))

    x_batch = np.hstack((
                         position_info_batch,
                         bid_info_batch,  # 3
                         bomb_num_batch,  # 15
                         ))
    x_no_action = np.hstack((
                             position_info,
                             bid_info,
                             bomb_num,
                             ))

    z = np.vstack((
                  num_cards_left,  # 54
                  my_handcards,  # 54
                  next_player_handcards,  # 54
                  nnext_player_handcards,  # 54
                  three_landlord_cards,  # 54
                  landlord_played_cards,  # 54
                  landlord_up_played_cards,  # 54
                  landlord_down_played_cards,  # 54
                  bid_info_z,
                  spring,
                  _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq, 60))
                  ))

    _z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    z_batch = np.concatenate((my_action_batch[:, np.newaxis, :], _z_batch), axis=1)
    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
        'legal_actions': infoset.legal_actions,
        # 'state': np.vstack((np.pad(x_no_action.astype(np.float32), (0, 54 - len(x_no_action))), z.astype(np.float32))),
    }
    
    return obs
    pass

def act(i, device, actor, batch_queues, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    
    try:
        T = flags.unroll_length
        print('(TractorsEnv)Device %s Actor %i started.', str(device), i)

        env = TractorsEnv(flags)
        # env = Environment(env, device)

        '''逐步迭代:
        1.从残局预测开始训练（可见信息最丰富），然后逐步加入更多的不可见信息。'''
        threshold_handcards = 5
        
        '''最终的回放buff，形状[T,15,4,2,4,15]，
        每个buff元素是一个形状为[15,4,2,4,15]的矩阵，15是轮数(history_play_cards)，后面是一轮的出牌'''
        history_play_card_buf = []
        history_play_seat_buf = []
        history_played_card_buf = []#本局已出牌
        #历史叫牌，形状[T,2,2,4,15]，只存2次，环境设置如此v
        
        
        #经验轨迹
        hand_cards_buf = []
        # obs['major_cards_buf'] = major_card_mtx
        # obs['public_cards_buf'] = public_cards
        # obs['played_score_cards_buf'] = played_score_cards
        # obs['remain_score_cards_buf'] = remain_score_cards
        # obs['history_play_cards_buf'] = history_play_cards
        # obs['round_play_cards_buf'] = round_play_cards
        # obs['last_play_cards_buf'] = last_play_cards
        # obs['played_cards_buf'] = played_cards
        # obs['mask_cards_buf'] = mask_cards
        # obs['my_seat_buf'] = get_one_hot_array(banker, __PLAYER_COUNT__)
        # obs['banker_seat_buf'] = get_one_hot_array(banker, __PLAYER_COUNT__)
        # obs['score_buf'] = get_full_hot_array(0, __MAX_SCORE__)
        # obs['win_score_buf'] = get_full_hot_array(bid_score//5, __MAX_SCORE__)
        
        

        while True:
            response = []
            
            #新一局开始
            env.reset()
            obs = {}
            
            #出牌阶段的回放经验

            #历史出牌序列信息
            history_play_cards = []
            history_play_seat = []

            #当前回合出牌序列信息            
            round_play_cards = []
            round_play_seat = []
            
            #最后一次出牌
            last_play_cards = []
            
            
            hand_cards = []


            ######################################################################################
            #报主缓存
            bid_trajectory = []
            bid_score = 0
            
            #出牌阶段的缓存            
            played_cards = []
            played_score_cards = []
            remain_score_cards = []

            #底牌缓存
            public_cards = None

            history_play_team = None
            major_cards = None
            
            #当前得分
            round_score = 0
            game_score = 0

            mask_cards = []
            ######################################################################################

            #回合出牌緩存
            round_cnt = 0
            play_counts = 0


            # round_player_remain_card_num = [0,0,0,0]
            record_index = False#是否开始记录轨迹
            inning_major = None
            inning_level = None
            #主牌
            major_cards_mtx = None
            
            infoset = {}

            
            while True:
                env.step(response)
                
                err = env.getError()
                if len(err)>0:
                    print(err[len(err)-1])
                    env.reset()
                    env.step(response)

                stage = env.getStage()
                #叫分阶段
                if stage == "bid":
                    play_pos = env.getPlayerPosition()                    
                    
                    if np.random.rand() < 0.5:
                        response = [play_pos, 0]
                    else:
                        bid_opt = math.max(0, (80/5 - len(bid_trajectory)))
                        response = [play_pos, random.randint(0, bid_opt*5)]
                        if response[1] > 0:
                            major_color = random.sample(__SUITSET__, 1)
                            response[2] = major_color
                            bid_score = response[1]
                    bid_trajectory.append(response)
                    
                    
                    # get_card = env.getDeliver()[0]
                    # called = env.getCalled()
                    # snatched = env.getSnatched()
                    # level = env.getLevel()
                    # play_pos = env.getPlayerPosition()                    
                    # hold = env.getPlayerHoldCards(play_pos)
                    # ret = env.call_Snatch(get_card, hold, called, snatched, level)
                    # response = [play_pos, ret]
                    # if len(ret) > 0:
                    #     bid_trajectory.append(response)
                    
                #埋牌阶段
                elif stage == "cover":
                    # cover_seat = response[0]
                    # cover_cards = response[1]
                    # major_color = response[2]
                    
                    banker = env.getBanker()
                    hold_cards = env.getPlayerHoldCards(banker)
                    cover_seat = banker                    
                    cover_cards = random.sample(hold_cards, 8)
                    response = [cover_seat, cover_cards]
                
                
                elif stage == 'startplay':
                    #self, public_cards, hold_card, own_seat, bid_history, level, major
                    # agent_output = actor.coverCard(publiccard, hold_cards, bid_trajectory, inning_major, inning_level)
                    
                    public_cards = cards2matrix(env.getPublicCards())
                                        
                    #桌面分
                    played_score_cards = cards2matrix([])
                    #隐藏信息分
                    remain_score_cards = [s + '5' for s in __SUITSET__] + [s + '0' for s in __SUITSET__] + [s + 'K' for s in __SUITSET__]
                    remain_score_cards = env.Pokers2Num(remain_score_cards,[i for i in range(54)])
                    remain_score_cards.extend([c+54 for c in remain_score_cards])
                    remain_score_cards = cards2matrix(remain_score_cards)
                    #已出牌
                    played_cards = [cards2matrix([]) for _ in range(__PLAYER_COUNT__)]
                    
                    #叫主轨迹信息
                    # history_bid_card = [cards2matrix([]) for _ in range(2)]
                    # history_bid_seat = [np.zeros(__PLAYER_COUNT__) for _ in range(2)]
                    # if len(bid_trajectory)>2:
                    #     KeyError('len(bid_trajectory)>2')
                    # for i,traj in enumerate(bid_trajectory):
                    #     history_bid_seat[i][traj[0]-1] = 1.0
                    #     history_bid_card[i] = cards2matrix(traj[1])
                    
                    #回合出牌序列信息
                    round_play_cards = [cards2matrix([]) for _ in range(__PLAYER_COUNT__)]
                    round_play_seat = [np.zeros(__PLAYER_COUNT__) for _ in range(__PLAYER_COUNT__)]
                    
                    #上回合出牌序列信息
                    last_play_cards = [cards2matrix([]) for _ in range(__PLAYER_COUNT__)]
                    
                    history_play_cards = [[cards2matrix([]) for _ in range(__PLAYER_COUNT__)] for _ in range(15)]
                    history_play_seat = [[np.zeros(__PLAYER_COUNT__) for _ in range(__PLAYER_COUNT__)] for _ in range(15)]

                    #手牌
                    hand_cards = [cards2matrix([env.getPlayerHoldCards(i)]) for i in range(__PLAYER_COUNT__)]
                    #mask隐蔽牌
                    mask_cards = [cards2matrix([1 for _ in range(__CARDS_NUM__)]) for _ in range(__PLAYER_COUNT__)]
                    #去掉自己的手牌
                    for i in range(len(mask_cards)):
                        mask_cards[i] = mask_cards[i] - hand_cards[i]
                    mask_cards[seat] = mask_cards[seat] - public_cards
                    
                    #极牌
                    major_card_mtx = cards2matrix(env.getMajorCards())

                    round_cnt = 0
                    play_counts = 0

                    #场面数据
                    # hand_cards 			[2,4,15]                #我的手牌
                    # major_cards			[2,4,15]                #历史级牌
                    # public_cards		[2,4,15]                    #底牌，只有banker可见
                    # played_score_cards	[2,4,15]                #已出分数牌
                    # remain_score_cards	[2,4,15]                #剩余分数牌
                    # history_play_cards	[15,PLAYER_COUNT,2,4,15]#历史出牌序列
                    # round_play_cards	[PLAYER_COUNT,2,4,15]       #当前回合出牌序列
                    # last_play_cards		[PLAYER_COUNT,2,4,15]   #上次回合出牌序列                    
                    # played_cards		[PLAYER_COUNT,2,4,15]       #已出牌
                    # mask_cards			[PLAYER_COUNT,2,4,15]   #当前玩家未知牌mask
                    # my_seat				[PLAYER_COUNT]          #我的座位号
                    # banker_seat			[PLAYER_COUNT]          #庄家座位号
                    # score				[40]                        #当前捡到的分数（1个占位为5分）
                    # win_score         [40]                        #赢的分数线
                    banker = env.getBanker()
                    infoset['hand_cards'] = hand_cards[banker]
                    infoset['major_cards'] = major_card_mtx
                    infoset['public_cards'] = public_cards
                    infoset['played_score_cards'] = played_score_cards
                    infoset['remain_score_cards'] = remain_score_cards
                    infoset['history_play_cards'] = history_play_cards
                    infoset['round_play_cards'] = round_play_cards
                    infoset['last_play_cards'] = last_play_cards
                    infoset['played_cards'] = played_cards
                    infoset['mask_cards'] = mask_cards
                    infoset['my_seat'] = get_one_hot_array(banker+1, __PLAYER_COUNT__)
                    infoset['play_rights_seat'] = get_one_hot_array(banker+1, __PLAYER_COUNT__)                    
                    infoset['banker_seat'] = get_one_hot_array(banker+1, __PLAYER_COUNT__)
                    infoset['score'] = get_full_hot_array(0, __MAX_SCORE__)
                    infoset['win_score'] = get_full_hot_array(bid_score//5, __MAX_SCORE__)
                    
                    
                #出牌阶段
                elif stage == "play":
                    play_pos = env.getPlayerPosition()
                    banker = env.getBanker()
                    
                    infoset['hand_cards'] = cards2matrix([env.getPlayerHoldCards(banker)])
                    infoset['my_seat'] = get_one_hot_array(banker+1, __PLAYER_COUNT__)
                    history_curr = env.getCurrRoundPlayHistory()
                    hold = env.getPlayerHoldCards(play_pos)
                    level = env.getLevel()
                    infoset['legal_actions'] = env.getLegalPlayCard(history_curr, hold, level)
                    infoset['num_cards_left'] = [env.getPlayerHoldCards(i)]
                    obs = get_obs(infoset)
                    
                    
                    #数据合法性验证 test code
                    # for trj in range(len(history_play_cards)):
                    #     count = 0
                    #     for seat in range(4):
                    #         count += np.sum(history_play_cards[trj][seat])
                    #     if count%2 != 0:
                    #         raise ValueError(history_play_cards[trj][seat])
                    
                    '''存储当前进度的经验回放 回合内的动态buf'''
                    if record_index:
                        obs['hand_cards'] = hand_cards[banker]
                        obs['major_cards'] = major_card_mtx
                        obs['public_cards'] = public_cards
                        obs['played_score_cards'] = played_score_cards
                        obs['remain_score_cards'] = remain_score_cards
                        obs['history_play_cards'] = history_play_cards
                        obs['round_play_cards'] = round_play_cards
                        obs['last_play_cards'] = last_play_cards
                        obs['played_cards'] = played_cards
                        obs['mask_cards'] = mask_cards
                        obs['my_seat'] = get_one_hot_array(banker+1, __PLAYER_COUNT__)
                        obs['banker_seat'] = get_one_hot_array(banker+1, __PLAYER_COUNT__)
                        obs['score'] = get_full_hot_array(0, __MAX_SCORE__)
                        obs['win_score'] = get_full_hot_array(bid_score//5, __MAX_SCORE__)
                        
                    
                    
                        mask_cards_buf.append(copy.deepcopy(mask_cards))
                        round_play_card_buf.append(copy.deepcopy(round_play_cards))
                        round_play_seat_buf.append(copy.deepcopy(round_play_seat))
                        history_play_card_buf.append(copy.deepcopy(history_play_cards))
                        history_play_seat_buf.append(copy.deepcopy(history_play_seat))
                        
                        played_score_card_buf.append(copy.deepcopy(played_score_cards))
                        remain_score_card_buf.append(copy.deepcopy(remain_score_cards))
                                                
                        hand_cards_buf.append(cards2matrix(env.getPlayerHoldCards(play_pos)))
                        
                        history_played_card_buf.append(copy.deepcopy(played_cards))
                        
                        #以下为牌局静态buf, 局内不变化                     
                        my_seat_buf.append(get_one_hot_array(play_pos+1, __PLAYER_COUNT__))
                        banker_seat_buf.append(get_one_hot_array(banker_pos+1, __PLAYER_COUNT__))
                        public_card_buf.append(np.copy(public_cards))
                        major_cards_buf.append(np.copy(major_card_mtx))
                        
                        #输赢距离
                        score = env.getTotalScore()
                        
                        # tractored = score >= 80
                        # win_index_buf.append(get_one_hot_array((1 if False == tractored else 2) if banker_pos == play_pos else (1 if True == tractored else 2), 2))
                        # win_ds_buf = []
                        
                   
                    #执行游戏出牌
                    history_curr = env.getCurrRoundPlayHistory()
                    hold = env.getPlayerHoldCards(play_pos)
                    playedCards = env.getLegalPlayCard(history_curr, hold, inning_level)
                    response = [play_pos, playedCards[random.randint(0, len(playedCards)-1)]]
                    playcard_mtrx = cards2matrix(response[1])
                    
                    '''更新回合动态buf'''
                    #mask隐蔽牌
                    for i in range(4):
                        mask_cards[i] = mask_cards[i] - playcard_mtrx
                    if play_counts == 0 :
                        frist_play_mtx = playcard_mtrx
                    else:
                        play_suit = np.any(playcard_mtrx[:, :, 0:-2]== 1, axis=(0,2))
                        play_suit_major = np.sum(np.any(playcard_mtrx[:, :, -3:]== 1, axis=(0,2)).astype(np.int32)) > 0
                        first_play_suit = np.any(frist_play_mtx[:, :, 0:-2] == 1, axis=(0,2))
                        first_play_suit_major = np.sum(np.any(frist_play_mtx[:, :, -3:]== 1, axis=(0,2)).astype(np.int32)) > 0

                        err = False
                        if first_play_suit_major or first_play_suit[0] == True:#首出是主
                            if play_suit_major == False and play_suit[0] == False:#跟牌不是主
                                mask_cards[play_pos][:, :, -3:] = 0#mask掉没有跟的主
                                #错误校验 test code
                                # handcard_mat = cards2matrix(hold)#看手牌里有没有可跟的牌
                                # hand_suit = np.any(handcard_mat[:, :, 0:-2]== 1, axis=(0,2))
                                # hand_suit_major = np.sum(np.any(handcard_mat[:, :, -3:]== 1, axis=(0,2)).astype(np.int32)) > 0
                                # if hand_suit_major == True and hand_suit[0] == True:
                                #     raise ValueError("出牌非法")

                        elif np.sum(play_suit == first_play_suit) != 4 and play_suit_major == False:#首出不是主，看是没是跟了牌
                            first_play_suit_where = np.where(first_play_suit == True)
                            if len(first_play_suit_where) != 1:
                                raise ValueError("first_play_suit_where非法")
                            mask_cards[play_pos][:, first_play_suit_where[0], 0:-3] = 0#mask掉没有跟的牌

                            # handcard_mat = cards2matrix(hold)
                            # handcard_suit = np.any(handcard_mat[:, :, 0:-2]== 1, axis=(0,2))
                            # handcard_suit_where = np.where(np.any(handcard_suit) == False)
                            
                            # if len(first_play_suit_where) != 1 or len(handcard_suit_where) != 1 or first_play_suit_where[0] != handcard_suit_where[0]:
                            #     raise ValueError("出牌非法")
                            
                    # 出牌序列
                    round_play_cards[play_counts] = playcard_mtrx
                    round_play_seat[play_counts][play_pos] = 1.0
                    
                    if play_counts >= len(history_play_cards):
                        while len(history_play_cards) >= 15:
                            history_play_cards.pop(0)
                            history_play_seat.pop(0)
                        history_play_cards.append(copy.deepcopy(round_play_cards))
                        history_play_seat.append(copy.deepcopy(round_play_seat))
                    else:
                        history_play_cards[play_counts] = copy.deepcopy(round_play_cards)
                        history_play_seat[play_counts] = copy.deepcopy(round_play_seat)
                    
                    
                    #分牌
                    play_score_card = playcard_mtrx * remain_score_cards
                    played_score_cards = played_score_cards + play_score_card
                    remain_score_cards = remain_score_cards - play_score_card
                        
                    #已出牌
                    played_cards = played_cards + playcard_mtrx    
                        
                            
                    

                    # if np.sum(playcard_mtrx[:,1:4, 13:14]) > 0:
                    #     raise ValueError("卡牌矩阵非法")
                    
                    # if np.sum(playcard_mtrx)%2 != 0 and np.sum(playcard_mtrx) != 1:
                    #     playedCards = env.getLegalPlayCard(history_curr, hold, inning_level)
                    #     raise ValueError("出牌报错，该出牌为空")
                    
                    
                    # for seat in range(play_counts):
                    #     if round_play_cards[seat].sum() != playcard_mtrx.sum():
                    #         raise ValueError("出牌报错，该出牌与历史出牌不一致")

                    

                    

                    
                    # # #test code 错误检验
                    # all_cards = played_score_cards + remain_score_cards
                    # card_num = np.sum(all_cards)
                    # if card_num != 24:
                    #     raise ValueError('分数牌不一致')
                    # for k in range(len(played_score_card_buf)):
                    #     score_cards = played_score_card_buf[k]
                    #     remain_score_cards = remain_score_card_buf[k]
                    #     all_cards = score_cards + remain_score_cards
                    #     card_num = np.sum(all_cards)
                    #     if card_num != 24:
                    #         raise ValueError('分数牌不一致')
                            
                    play_counts += 1
                    

                #一回合结束
                elif stage == 'roundend' or stage == 'gameend':
                    if len(env.getPlayerHoldCards(env.getPlayerPosition())) <= threshold_handcards:
                        record_index = True
                    
                    round_score = env.getLastRoundScore()
                    game_score = env.getTotalScore()
                    
                    if stage == 'roundend':
                        banker = env.getBanker()
                        for seat in range(__PLAYER_COUNT__):
                            win = 1 if seat == banker or (seat + 2)%__PLAYER_COUNT__ == banker else -1
                            reward_buf.append(np.array([win*game_score[seat]], dtype=np.float32))
                            
                            
                        #存储训练用的经验回放 即时奖励
                        while len(mask_cards_buf) > T:
                            batch_queues.put({
                                "episode_return": torch.stack(
                                    [torch.tensor(ndarr, device="cpu") for ndarr in episode_return_buf[p][:T]]),
                                "target_adp": torch.stack(
                                    [torch.tensor(ndarr, device="cpu") for ndarr in target_adp_buf[p][:T]]),
                                "target_wp": torch.stack(
                                    [torch.tensor(ndarr, device="cpu") for ndarr in target_wp_buf[p][:T]]),
                                "target_wp_bid": torch.stack(
                                    [ndarr.clone().detach() for ndarr in target_wp_bid_buf[p][:T]]),
                                "obs_z": torch.stack([ndarr.clone().detach() for ndarr in obs_z_buf[p][:T]]),
                                "obs_x_batch": torch.stack(
                                    [ndarr.clone().detach() for ndarr in obs_x_batch_buf[p][:T]]),
                            })
                            done_buf[p] = done_buf[p][T:]
                            episode_return_buf[p] = episode_return_buf[p][T:]
                            target_adp_buf[p] = target_adp_buf[p][T:]
                            target_wp_buf[p] = target_wp_buf[p][T:]
                            target_wp_bid_buf[p] = target_wp_bid_buf[p][T:]
                            obs_x_batch_buf[p] = obs_x_batch_buf[p][T:]
                            obs_z_buf[p] = obs_z_buf[p][T:]
                            size[p] -= T
                            
                            index = free_queue.get()#bug 这里容易卡死 batchszie要小于num_buffers，batchszie不够就会一直等待足够的num_buffers，num_buffers又会等待batchsize训练数据释放
                            if index is None:
                                break
                            for t in range(T):
                                # buffers是tensor history_play_card_buff是array
                                buffers['history_play_cards'][index][t, ...] = torch.tensor(history_play_card_buf[t])
                                buffers['history_play_seat'][index][t, ...] = torch.tensor(history_play_seat_buf[t])
                                buffers['played_cards'][index][t, ...] = torch.tensor(history_played_card_buf[t])
                                buffers['history_bid_card'][index][t, ...] = torch.tensor(history_bid_card_buf[t])
                                buffers['history_bid_seat'][index][t, ...] = torch.tensor(history_bid_seat_buf[t])
                                buffers['round_play_cards'][index][t, ...] = torch.tensor(round_play_card_buf[t])
                                buffers['round_play_seat'][index][t, ...] = torch.tensor(round_play_seat_buf[t])
                                buffers['score_card'][index][t, ...] = torch.tensor(played_score_card_buf[t])
                                buffers['remain_score_cards'][index][t, ...] = torch.tensor(remain_score_card_buf[t])
                                buffers['my_seat'][index][t, ...] = torch.tensor(my_seat_buf[t])
                                buffers['banker_seat'][index][t, ...] = torch.tensor(banker_seat_buf[t])
                                buffers['public_cards'][index][t, ...] = torch.tensor(public_card_buf[t])
                                buffers['hand_card'][index][t, ...] = torch.tensor(hand_cards_buf[t])
                                buffers['mask_card'][index][t, ...] = torch.tensor(mask_cards_buf[t])#'''特征工程 规则层特征'''    
                                
                            
                            full_queue.put(index)
                            history_play_card_buf = history_play_card_buf[T:]
                            history_play_seat_buf = history_play_seat_buf[T:]
                            history_played_card_buf = history_played_card_buf[T:]
                            history_bid_card_buf = history_bid_card_buf[T:]
                            history_bid_seat_buf = history_bid_seat_buf[T:]
                            round_play_card_buf = round_play_card_buf[T:]
                            round_play_seat_buf = round_play_seat_buf[T:]
                            played_score_card_buf = played_score_card_buf[T:]
                            remain_score_card_buf = remain_score_card_buf[T:]
                            my_seat_buf = my_seat_buf[T:]
                            banker_seat_buf = banker_seat_buf[T:]
                            public_card_buf = public_card_buf[T:]
                            hand_cards_buf = hand_cards_buf[T:]
                            mask_cards_buf = mask_cards_buf[T:]
                    
                    # #数据合法性验证 test code
                    # for trj in range(len(history_play_cards)):
                    #     count = 0
                    #     for seat in range(4):
                    #         count += np.sum(history_play_cards[trj][seat])
                    #     if count%2 != 0:
                    #         raise ValueError(history_play_cards[trj][seat])
                    
                        

                    #重置回合信息
                    round_play_cards = [cards2matrix([]) for _ in range(__PLAYER_COUNT__)]
                    round_play_seat = [np.zeros(__PLAYER_COUNT__) for _ in range(__PLAYER_COUNT__)]                    
                    if stage == 'roundend': 
                        play_counts = 0
                        round_cnt += 1
                        
                    response = None
                    
                    
                    
                elif stage == "finalend":
                    break
            
            
            

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e