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
import os
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



def run(i, device, actor, free_queue, full_queue, buffers, flags):
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

        #发牌阶段的回放经验
        bid_card_buff = []
        bid_seat_buff = []
        
        '''逐步迭代:
        1.从残局预测开始训练（可见信息最丰富），然后逐步加入更多的不可见信息。'''
        threshold_handcards = 5
        
        '''最终的回放buff，形状[T,15,4,2,4,15]，
        每个buff元素是一个形状为[15,4,2,4,15]的矩阵，15是轮数(history_play_card)，后面是一轮的出牌'''
        history_play_card_buff = []
        history_play_seat_buff = []
        history_played_card_buff = []#本局已出牌
        #历史叫牌，形状[T,2,2,4,15]，只存2次，环境设置如此v
        history_bid_card_buff = []        
        history_bid_seat_buff = []
        #本轮出牌信息，形状[T,4,2,4,15]
        round_play_card_buff = []
        round_play_seat_buff = []
        score_card_buff = []
        remain_score_card_buff = []
        my_seat_buff = []
        banker_seat_buff = []
        #公共牌缓存
        public_card_buff = []
        #手牌缓存
        hand_cards_buff = []

        while True:
            response = []
            
            #新一局开始
            env.reset()
            obs_x = {}
            
            #出牌阶段的回放经验

            
            history_play_card = []#15轮的出牌信息
            history_play_seat = []

            

            
            
            
            round_play_card = []
            round_play_seat = []

            
            
            

            ######################################################################################
            #报主缓存
            bid_trajectory = []
            history_bid_card = []
            history_bid_seat = []

            

            #出牌阶段的缓存
            
            history_played_card = []
            history_score_card = []
            history_remain_score_card = []

            #底牌缓存
            history_public_card = None

            history_play_team = None
            history_level_card = None
            ######################################################################################

            #回合出牌緩存
            round_cnt = 0
            round_times = 0


            # round_player_remain_card_num = [0,0,0,0]
            record_traj = False#是否开始记录轨迹
            inning_major = None
            inning_level = None

            
            while True:
                env.step(response)
                
                err = env.getError()
                if len(err)>0:
                    print(err[len(err)-1])
                    env.reset()
                    env.step(response)

                stage = env.getStage()
                #发牌阶段
                if stage == "deal":
                    get_card = env.getDeliver()[0]
                    called = env.getCalled()
                    snatched = env.getSnatched()
                    level = env.getLevel()
                    play_pos = env.getPlayerPosition()
                    
                    hold = env.getPlayerHoldCards(play_pos)
                    ret = env.call_Snatch(get_card, hold, called, snatched, level)
                    response = [play_pos, ret]
                    if len(ret) > 0:
                        bid_trajectory.append(response)
                    
                    

                #埋牌阶段
                elif stage == "cover":
                    publiccard = env.getPublicCards()
                    banker = env.getBanker()
                    seat = env.getPlayerPosition()
                    hold_cards = env.getPlayerHoldCards(banker)
                    inning_major = env.getMajor()
                    inning_level = env.getLevel()
                    
                    #self, public_card, hold_card, own_seat, bid_history, level, major
                    # agent_output = actor.coverCard(publiccard, hold_cards, bid_trajectory, inning_major, inning_level)
                    agent_output = env.cover_PubEx(publiccard, hold_cards, inning_level)
                    response = [banker, agent_output]
                    history_public_card = cards2matrix(response[1], inning_level, inning_major)
                    
                    obs_x = {}

                    history_level_card = cards2matrix([], inning_level, inning_major)
                    history_level_card[:,0,:] = 1
                    history_level_card[:,:,12] = 1                    
                    obs_x['level_card'] = history_level_card#历史级牌
                    

                    #桌面分
                    history_score_card = cards2matrix([], inning_level, inning_major)
                    #隐藏信息分
                    history_remain_score_card = [s + '5' for s in __SUITSET__] + [s + '0' for s in __SUITSET__] + [s + 'K' for s in __SUITSET__]
                    history_remain_score_card = env.Pokers2Num(history_remain_score_card,[i for i in range(54)])
                    history_remain_score_card.extend([c+54 for c in history_remain_score_card])
                    history_remain_score_card = cards2matrix(history_remain_score_card, inning_level, inning_major)
                    #已出牌
                    history_played_card = cards2matrix([], inning_level, inning_major)
                    #叫主轨迹信息
                    history_bid_card = [cards2matrix([], inning_level, inning_major) for _ in range(2)]
                    history_bid_seat = [np.zeros(__PLAYER_COUNT__) for _ in range(2)]
                    if len(bid_trajectory)>2:
                        KeyError('len(bid_trajectory)>2')
                    for i,traj in enumerate(bid_trajectory):
                        history_bid_seat[i][traj[0]-1] = 1.0
                        history_bid_card[i] = cards2matrix(traj[1], inning_level, inning_major)
                    
                    round_play_card = [cards2matrix([], inning_level, inning_major) for _ in range(__PLAYER_COUNT__)]
                    round_play_seat = [np.zeros(__PLAYER_COUNT__) for _ in range(__PLAYER_COUNT__)]
                    history_play_card = [[cards2matrix([], inning_level, inning_major) for _ in range(__PLAYER_COUNT__)] for _ in range(15)]
                    history_play_seat = [[np.zeros(__PLAYER_COUNT__) for _ in range(__PLAYER_COUNT__)] for _ in range(15)]
                    round_times = 0
                
                    # obs_x['public_card'] = cards2matrix(agent_output, inning_level, inning_major)
                    # obs_x['history_play_card'] = []
                    # obs_x['history_play_seat'] = []
                    # obs_x['played_card'] = cards2matrix([], inning_level, inning_major)#历史所有已出牌
                    # obs_x['history_bid_card'] = []
                    # obs_x['history_bid_seat'] = []
                    # obs_x['score_card'] = history_score_card#历史分得分牌
                    # obs_x['remain_score_card'] = cards2matrix(history_remain_score_card, inning_level, inning_major)#历史剩余分数牌
                    # obs_x['hand_card'] = []
                    
                    # for bid_traj in bid_trajectory:
                    #     obs_x['history_bid_card'].append(cards2matrix(bid_traj[1], inning_level, inning_major))
                    #     obs_x['history_bid_seat'].append(np.zeros(__PLAYER_COUNT__))
                    #     obs_x['history_bid_seat'][len(obs_x['history_bid_seat'])-1][bid_traj[0]-1] = 1


                #出牌阶段
                elif stage == "play":
                    play_pos = env.getPlayerPosition()
                    banker_pos = env.getBanker()

                    #数据合法性验证
                    for trj in range(len(history_play_card)):
                        count = 0
                        for seat in range(4):
                            count += np.sum(history_play_card[trj][seat])
                        if count%2 != 0:
                            raise ValueError(history_play_card[trj][seat])
                                
                    if record_traj: 
                        #存储当前进度的经验回放
                        history_play_card_buff.append(copy.deepcopy(history_play_card))
                        history_play_seat_buff.append(copy.deepcopy(history_play_seat))
                        history_played_card_buff.append(copy.deepcopy(history_played_card))
                        history_bid_card_buff.append(copy.deepcopy(history_bid_card))
                        history_bid_seat_buff.append(copy.deepcopy(history_bid_seat))
                        round_play_card_buff.append(copy.deepcopy(round_play_card))
                        round_play_seat_buff.append(copy.deepcopy(round_play_seat))
                        score_card_buff.append(copy.deepcopy(history_score_card))
                        remain_score_card_buff.append(copy.deepcopy(history_remain_score_card))
                        my_seat_buff.append(get_one_hot_array(play_pos+1, __PLAYER_COUNT__))
                        banker_seat_buff.append(get_one_hot_array(banker_pos+1, __PLAYER_COUNT__))
                        public_card_buff.append(history_public_card)
                        hand_cards_buff.append([cards2matrix(env.getPlayerHoldCards(seat), inning_level, inning_major) for seat in range(__PLAYER_COUNT__)])
                        
                        #存储训练用的经验回放
                        while len(history_play_card_buff) > T:
                            index = free_queue.get()#bug 这里容易卡死 batchszie要小于num_buffers，batchszie不够就会一直等待足够的num_buffers，num_buffers又会等待batchsize训练数据释放
                            if index is None:
                                break
                            for t in range(T):
                                # buffers是tensor history_play_card_buff是array
                                buffers['history_play_card'][index][t, ...] = torch.tensor(history_play_card_buff[t])
                                buffers['history_play_seat'][index][t, ...] = torch.tensor(history_play_seat_buff[t])
                                buffers['history_played_card'][index][t, ...] = torch.tensor(history_played_card_buff[t])
                                buffers['history_bid_card'][index][t, ...] = torch.tensor(history_bid_card_buff[t])
                                buffers['history_bid_seat'][index][t, ...] = torch.tensor(history_bid_seat_buff[t])
                                buffers['round_play_card'][index][t, ...] = torch.tensor(round_play_card_buff[t])
                                buffers['round_play_seat'][index][t, ...] = torch.tensor(round_play_seat_buff[t])
                                buffers['score_card'][index][t, ...] = torch.tensor(score_card_buff[t])
                                buffers['remain_score_card'][index][t, ...] = torch.tensor(remain_score_card_buff[t])
                                buffers['my_seat'][index][t, ...] = torch.tensor(my_seat_buff[t])
                                buffers['banker_seat'][index][t, ...] = torch.tensor(banker_seat_buff[t])
                                buffers['public_card'][index][t, ...] = torch.tensor(public_card_buff[t])
                                buffers['hand_card'][index][t, ...] = torch.tensor(hand_cards_buff[t])
                                
                            
                            full_queue.put(index)
                            history_play_card_buff = history_play_card_buff[T:]
                            history_play_seat_buff = history_play_seat_buff[T:]
                            history_played_card_buff = history_played_card_buff[T:]
                            history_bid_card_buff = history_bid_card_buff[T:]
                            history_bid_seat_buff = history_bid_seat_buff[T:]
                            round_play_card_buff = round_play_card_buff[T:]
                            round_play_seat_buff = round_play_seat_buff[T:]
                            score_card_buff = score_card_buff[T:]
                            remain_score_card_buff = remain_score_card_buff[T:]
                            my_seat_buff = my_seat_buff[T:]
                            banker_seat_buff = banker_seat_buff[T:]
                            public_card_buff = public_card_buff[T:]
                            hand_cards_buff = hand_cards_buff[T:]
                    
                    #执行游戏出牌
                    history_curr = env.getCurrRoundPlayHistory()
                    hold = env.getPlayerHoldCards(play_pos)
                    playedCards = env.getLegalPlayCard(history_curr, hold, inning_level)
                    response = [play_pos, playedCards[random.randint(0, len(playedCards)-1)]]
                    playcard_mtrx = cards2matrix(response[1], inning_level, inning_major)
                    # if np.sum(playcard_mtrx)%2 != 0 and np.sum(playcard_mtrx) != 1:
                    #     playedCards = env.getLegalPlayCard(history_curr, hold, inning_level)
                    #     raise ValueError("出牌报错，该出牌为空")
                    
                    
                    # print('playcard_mtrx.sum()', playcard_mtrx.sum(), history_curr)
                    if record_traj: 
                        history_played_card = history_played_card + playcard_mtrx
                        for seat in range(round_times):
                            if round_play_card[seat].sum() != playcard_mtrx.sum():
                                raise ValueError("出牌报错，该出牌与历史出牌不一致")

                        # 回合内信息
                        round_play_card[round_times] = playcard_mtrx
                        round_play_seat[round_times][play_pos-1] = 1.0

                        #分牌
                        play_score_card = playcard_mtrx * history_remain_score_card
                        history_score_card = history_score_card + play_score_card
                        history_remain_score_card = history_remain_score_card - play_score_card

                        # #test code 错误检验
                        all_cards = history_score_card + history_remain_score_card
                        card_num = np.sum(all_cards)
                        if card_num != 24:
                            raise ValueError('分数牌不一致')
                        

                        for k in range(len(score_card_buff)):
                            score_cards = score_card_buff[k]
                            remain_score_cards = remain_score_card_buff[k]
                            all_cards = score_cards + remain_score_cards
                            card_num = np.sum(all_cards)
                            if card_num != 24:
                                raise ValueError('分数牌不一致')
                            
                    round_times += 1
                    

                #一回合结束
                elif stage == 'roundend' or stage == 'gameend':
                    # 把当前回合历史信息添加到历史信息buff中
                    # 把当前回合历史信息添加到历史信息buff中
                    if round_cnt >= len(history_play_card):
                        # 如果round_cnt超过数组长度，删除前面的元素，保持数组大小为15
                        while len(history_play_card) >= 15:
                            history_play_card.pop(0)
                            history_play_seat.pop(0)
                        # 添加新的位置
                        history_play_card.append(copy.deepcopy(round_play_card))
                        history_play_seat.append(copy.deepcopy(round_play_seat))
                    else:
                        history_play_card[round_cnt] = copy.deepcopy(round_play_card)
                        history_play_seat[round_cnt] = copy.deepcopy(round_play_seat)
                    
                    #数据合法性验证 test code
                    for trj in range(len(history_play_card)):
                        count = 0
                        for seat in range(4):
                            count += np.sum(history_play_card[trj][seat])
                        if count%2 != 0:
                            raise ValueError(history_play_card[trj][seat])
                    
                        

                    #重置回合信息
                    round_play_card = [cards2matrix([], inning_level, inning_major) for _ in range(__PLAYER_COUNT__)]
                    round_play_seat = [np.zeros(__PLAYER_COUNT__) for _ in range(__PLAYER_COUNT__)]                    
                    if stage == 'roundend': 
                        round_times = 0
                        round_cnt += 1
                        
                    response = None
                    
                    if len(env.getPlayerHoldCards(env.getPlayerPosition())) <= threshold_handcards:
                        record_traj = True

                    #结束
                    if stage == 'gameend':
                        # cover_reward_buff.append(reward)
                        break
            
            

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e