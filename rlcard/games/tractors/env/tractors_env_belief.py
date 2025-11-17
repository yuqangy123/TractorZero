from collections import Counter, OrderedDict, deque
import numpy as np
import torch
from itertools import chain
from rlcard.games.tractors.env import tractors_game as Game
from rlcard.games.tractors.utils import *
import logging, random
import traceback

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('tractors_env_belief')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

class TractorsEnvBelief(Game):
    ''' tractor Environment
    '''

    def __init__(self, config):
        # from rlcard.games.doudizhu.utils import ACTION_2_ID, ID_2_ACTION
        # from rlcard.games.doudizhu.utils import cards2str, cards2str_with_suit
        
        # self._cards2str = cards2str
        # self._cards2str_with_suit = cards2str_with_suit
        # self._ACTION_2_ID = ACTION_2_ID
        # self._ID_2_ACTION = ID_2_ACTION
        super().__init__(config)

        # self.state_shape = [[790], [901], [901]]
        # self.action_shape = [[54] for _ in range(self.num_players)]

    def step(self, resp):
        super().step(resp)
        # s = self.getStage()
        # if s == 'deal':
            
    #亮牌竞价，并返回状态信息
    def bidding(self, seat):
        assert 1 <= seat <= __PLAYER_COUNT__
        level = self.getLevel()
        get_card = self.getDeliver()[0]
        called = self.getCalled()
        snatched = self.getSnatched()
        hold_card = self.getPlayerHoldCards(seat)
        bid_pok = self.call_Snatch(get_card, hold_card, called, snatched, level)
        if len(bid_pok) > 0:
            hold_card = get_card+hold_card
            #手牌, 竞价亮的牌， 亮牌历史（自己位置是第一顺位），  自己待发牌张数， 未发牌
            hold_card_feature = cards2matrix(hold_card, level)

            # major_card = cards2matrix([major_card])
            
            called_array = np.zeros((4,4))
            called_array[0,owned-1] = 1
            for i in range(len(called_list)):
                called_array[i+1,called_list[i]-1] = 1
                
            num_cards_left = get_one_hot_array(25-len(hold_card), 25)

    
    def _extract_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state
        '''
        current_hand = _cards2array(state['current_hand'])
        others_hand = _cards2array(state['others_hand'])

        last_action = ''
        if len(state['trace']) != 0:
            if state['trace'][-1][1] == 'pass':
                last_action = state['trace'][-2][1]
            else:
                last_action = state['trace'][-1][1]
        last_action = _cards2array(last_action)

        last_9_actions = _action_seq2array(_process_action_seq(state['trace']))

        if state['self'] == 0: # landlord
            landlord_up_played_cards = _cards2array(state['played_cards'][2])
            landlord_down_played_cards = _cards2array(state['played_cards'][1])
            landlord_up_num_cards_left = _get_one_hot_array(state['num_cards_left'][2], 17) 
            landlord_down_num_cards_left = _get_one_hot_array(state['num_cards_left'][1], 17)
            obs = np.concatenate((current_hand,
                                  others_hand,
                                  last_action,
                                  last_9_actions,
                                  landlord_up_played_cards,
                                  landlord_down_played_cards,
                                  landlord_up_num_cards_left,
                                  landlord_down_num_cards_left))
        else:
            landlord_played_cards = _cards2array(state['played_cards'][0])
            for i, action in reversed(state['trace']):
                if i == 0:
                    last_landlord_action = action
                    break
            last_landlord_action = _cards2array(last_landlord_action)
            landlord_num_cards_left = _get_one_hot_array(state['num_cards_left'][0], 20)

            teammate_id = 3 - state['self']
            teammate_played_cards = _cards2array(state['played_cards'][teammate_id])
            last_teammate_action = 'pass'
            for i, action in reversed(state['trace']):
                if i == teammate_id:
                    last_teammate_action = action
                    break
            last_teammate_action = _cards2array(last_teammate_action)
            teammate_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_id], 17)
            obs = np.concatenate((current_hand,
                                  others_hand,
                                  last_action,
                                  last_9_actions,
                                  landlord_played_cards,
                                  teammate_played_cards,
                                  last_landlord_action,
                                  last_teammate_action,
                                  landlord_num_cards_left,
                                  teammate_num_cards_left))

        extracted_state = OrderedDict({'obs': obs, 'legal_actions': self._get_legal_actions()})
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['actions']]
        extracted_state['action_record'] = self.action_recorder
        return extracted_state
            
    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        return self.game.judger.judge_payoffs(self.game.round.landlord_id, self.game.winner_id)

    def _decode_action(self, action_id):
        ''' Action id -> the action in the game. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action (string): the action that will be passed to the game engine.
        '''
        return self._ID_2_ACTION[action_id]

    def _get_legal_actions(self):
        ''' Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        legal_actions = self.game.state['actions']
        legal_actions = {self._ACTION_2_ID[action]: _cards2array(action) for action in legal_actions}
        return legal_actions

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['hand_cards_with_suit'] = [self._cards2str_with_suit(player.current_hand) for player in self.game.players]
        state['hand_cards'] = [self._cards2str(player.current_hand) for player in self.game.players]
        state['trace'] = self.game.state['trace']
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.state['actions']
        return state

    def get_action_feature(self, action):
        ''' For some environments such as DouDizhu, we can have action features

        Returns:
            (numpy.array): The action features
        '''
        return _cards2array(self._decode_action(action))

Card2Column = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7,
               'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

def _cards2array(cards):
    if cards == 'pass':
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(cards)
    for card, num_times in counter.items():
        if card == 'B':
            jokers[0] = 1
        elif card == 'R':
            jokers[1] = 1
        else:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
    return np.concatenate((matrix.flatten('F'), jokers))

def _get_one_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards, dtype=np.int8)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _action_seq2array(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 54), np.int8)
    for row, cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(cards)
    action_seq_array = action_seq_array.flatten()
    return action_seq_array

def _process_action_seq(sequence, length=9):
    sequence = [action[1] for action in sequence[-length:]]
    if len(sequence) < length:
        empty_sequence = ['' for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


def run(i, device, actor, free_queue, full_queue, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    
    try:
        T = flags.unroll_length
        print('(TractorsEnvBelief)Device %s Actor %i started.', str(device), i)

        env = TractorsEnvBelief(flags)
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
                    history_remain_score_card = env.Pokers2Num(history_remain_score_card,[])
                    history_remain_score_card.extend([c+54 for c in history_remain_score_card])
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