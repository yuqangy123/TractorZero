from collections import Counter, OrderedDict, deque
import numpy as np
import torch
from itertools import chain
from rlcard.games.tractors.utils import *
import logging, random
import traceback
from rlcard.envs import Env
from rlcard.games.tractors.tractor_botzone import tractorGame as tractors

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
        

        #出牌阶段的回放经验
        history_play_card_buff = []
        history_play_seat_buff = []
        history_played_card_buff = []
        history_bid_card_buff = []        
        history_bid_seat_buff = []
        round_play_card_buff = []
        round_play_seat_buff = []

        score_card_buff = []
        remain_score_card_buff = []
        my_seat_buff = []
        
        

        ######################################################################################
        #报主缓存
        bid_trajectory = []
        history_bid_card = []
        history_bid_seat = []

        #公共牌缓存
        public_card_buff = []
        #手牌缓存
        hand_cards_buff = []

        #出牌阶段的缓存
        history_play_card = []
        history_play_seat = []
        history_played_card = []
        history_score_card = []
        history_remain_score_card = []

        #底牌缓存
        history_public_card = None

        history_play_team = None
        history_level_card = None
        ######################################################################################

        #回合出牌緩存
        round_play_card = []
        round_play_seat = []


        # round_player_remain_card_num = [0,0,0,0]

        


        inning_major = None
        inning_level = None



        while True:
            response = []
            
            #新一局开始
            env.reset()
            obs_x = {}
            
            
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
                    history_bid_card = []
                    history_bid_seat = []
                    for traj in bid_trajectory:
                        pos_mat = np.zeros(__PLAYER_COUNT__)
                        pos_mat[traj[0]-1] = 1
                        history_bid_seat.append(pos_mat)
                        history_bid_card.append(cards2matrix[traj[1], inning_level, inning_major])
                        
                    

                    
                    obs_x['public_card'] = cards2matrix(agent_output, inning_level, inning_major)
                    obs_x['history_play_card'] = []
                    obs_x['history_play_seat'] = []
                    obs_x['played_card'] = cards2matrix([], inning_level, inning_major)#历史所有已出牌
                    obs_x['history_bid_card'] = []
                    obs_x['history_bid_seat'] = []
                    obs_x['score_card'] = history_score_card#历史分得分牌
                    obs_x['remain_score_card'] = cards2matrix(history_remain_score_card, inning_level, inning_major)#历史剩余分数牌
                    obs_x['hand_card'] = []
                    
                    for bid_traj in bid_trajectory:
                        obs_x['history_bid_card'].append(cards2matrix(bid_traj[1], inning_level, inning_major))
                        obs_x['history_bid_seat'].append(np.zeros(__PLAYER_COUNT__))
                        obs_x['history_bid_seat'][len(obs_x['history_bid_seat'])-1][bid_traj[0]-1] = 1


                #出牌阶段
                elif stage == "play":
                    play_pos = env.getPlayerPosition()
                    

                    #存储经验回放
                    history_play_card_buff.append(np.array(history_play_card))
                    history_play_seat_buff.append(np.array(history_play_seat))
                    history_played_card_buff.append(history_played_card)
                    history_bid_card_buff.append(history_bid_card)
                    history_bid_seat_buff.append(history_bid_seat)
                    round_play_card_buff.append(round_play_card)
                    round_play_seat_buff.append(round_play_seat)
                    score_card_buff.append(history_score_card)
                    remain_score_card_buff.append(history_remain_score_card)
                    my_seat_buff.append(play_pos)
                    public_card_buff.append(history_public_card)
                    hand_cards_buff.append([cards2matrix(env.getPlayerHoldCards(seat), inning_level, inning_major) for seat in range(__PLAYER_COUNT__)])

                    
                    #执行游戏出牌
                    history_curr = env.getCurrRoundPlayHistory()
                    hold = env.getPlayerHoldCards(play_pos)
                    playedCards = env.getLegalPlayCard(history_curr, hold, inning_level)
                    response = [play_pos, playedCards[random.randint(0, len(playedCards)-1)]]

                    

                    playcard_mtrx = cards2matrix(response[1], inning_level, inning_major)
                    #局内累计信息
                    history_play_card.append(playcard_mtrx)
                    history_play_seat.append(np.zeros(__PLAYER_COUNT__))
                    history_play_seat[len(history_play_seat)-1][play_pos-1] = 1
                    history_played_card += playcard_mtrx

                    # 回合内信息
                    round_play_card.append(playcard_mtrx)
                    round_play_seat.append(np.zeros(__PLAYER_COUNT__))
                    round_play_seat[len(round_play_seat)-1][play_pos-1] = 1

                    #分牌
                    play_score_card = playcard_mtrx * history_remain_score_card
                    history_score_card += play_score_card
                    history_remain_score_card -= play_score_card



                #一回合结束
                elif stage == 'roundend' or stage == 'gameend':
                    round_play_card = []
                    round_play_seat = []

                    #存储经验回放
                    while len(hand_cards_buff) > T:
                        index = free_queue.get()
                        if index is None:
                            break
                        for t in range(T):
                            # buffers是tensor history_play_card_buff是array
                            buffers['history_play_card'][index][t, ...] = history_play_card_buff[t]
                            buffers['history_play_seat'][index][t, ...] = history_play_seat_buff[t]
                            buffers['history_played_card'][index][t, ...] = history_played_card_buff[t]
                            buffers['history_bid_card'][index][t, ...] = history_bid_card_buff[t]
                            buffers['history_bid_seat'][index][t, ...] = history_bid_seat_buff[t]
                            buffers['round_play_card'][index][t, ...] = round_play_card_buff[t]
                            buffers['round_play_seat'][index][t, ...] = round_play_seat_buff[t]
                            buffers['score_card'][index][t, ...] = score_card_buff[t]
                            buffers['remain_score_card'][index][t, ...] = remain_score_card_buff[t]
                            buffers['my_seat'][index][t, ...] = my_seat_buff[t]
                            buffers['public_card'][index][t, ...] = public_card_buff[t]
                            buffers['hand_card'][index][t, ...] = hand_cards_buff[t]
                            
                        full_queue.put(index)
                        history_play_card_buff = history_play_card_buff[T:]
                        history_play_seat_buff = history_play_seat_buff[T:]
                        history_played_card_buff = history_played_card_buff[T:]
                        history_bid_card_buff = history_bid_card_buff[T:]
                        history_bid_seat_buff = history_bid_seat_buff[T:]
                        round_play_card_buff = round_play_card_buff[T:]
                        round_play_seat_buff = round_play_seat_buff[T:]
                        score_card_buff = score_card_buff[T:]
                        my_seat_buff = my_seat_buff[T:]
                        public_card_buff = public_card_buff[T:]
                        hand_cards_buff = hand_cards_buff[T:]
                        
                    response = None
                    
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