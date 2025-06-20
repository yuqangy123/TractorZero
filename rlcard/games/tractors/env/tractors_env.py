from collections import Counter, OrderedDict, deque
import numpy as np
import torch
from itertools import chain
from rlcard.games.tractors.env import tractors_game as Game
from rlcard.games.tractors.utils import *


class TractorsEnv(Game):
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


def run(i, device, free_queue, full_queue, actor, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['bid', 'conver', 'banker', 'player']
    try:
        T = flags.unroll_length
        print('Device %s Actor %i started.', str(device), i)

        env = TractorsEnv(flags)
        # env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        #发牌阶段的回放经验
        bid_buff = []
        inning_bid_buff = []
        #埋牌阶段的回放经验
        cover_buff = []
        inning_cover_buff = []
        #出牌阶段的回放经验
        history_play_card_buff = []
        history_play_seat_buff = []
        history_play_team_buff = []
        seat_buff = []
        hand_cards_buff = []
        round_play_card_buff = []
        round_play_seat_buff = []
        round_play_team_buff = []
        played_card_buff = []
        level_card_buff = []
        score_card_buff = []
        remain_score_card_buff = []
        reward_buff = []
        banker_buff = []
        
        #出牌阶段的缓存
        history_play_card = None
        history_play_seat = None
        history_play_team = None
         

        input_x = [{},{},{},{}]

        round_play_seat = [0,0,0]
        round_play_card = [0,0,0]
        round_play_team = [0,0,0]

        inning_major = None
        inning_level = None

        mat_level_card = None#极牌的矩阵
        mat_score_card = None#当前捡分的牌的矩阵
        mat_remain_score_card = None#剩余分的牌的矩阵
        mat_played_card = None#当前局已出的牌的矩阵

        

        while True:
            response = []
            
            #新一局开始
            env.reset()
            
            
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
                    level = env.getLevel()
                    deal_card = env.getDeliver()[0]
                    called_list = env.getCalledList()       
                    own_pos = env.getPlayerPosition()
                    hold_cards = env.getPlayerHoldCards(own_pos)
                    major = env.getMajor()

                    with torch.no_grad():
                        #self, get_card, hold_card, bid_card, own_seat, bid_history, major, level
                        bid_card = [env.Poker2Num(major + level,[])]*len(called_list)#构造当前已经叫的主牌
                        agent_output = actor.biddingMajor(deal_card, hold_cards, bid_card, own_pos, called_list, major, level)
                    response = [own_pos, agent_output]
                    if len(agent_output)>0:
                        inning_bid_buff = [level, deal_card, called_list, own_pos, hold_cards, major, agent_output]
                   


                #埋牌阶段
                elif stage == "cover":
                    publiccard = env.getPublicCards()
                    banker = env.getBanker()
                    hold_cards = env.getPlayerHoldCards(banker)
                    inning_major = env.getMajor()
                    inning_level = env.getLevel()
                    #self, public_card, hold_card, own_seat, bid_history, level, major
                    agent_output = actor.coverCard(publiccard, hold_cards, inning_major, inning_level)
                    response = [banker, agent_output]
                    if len(agent_output)>0:
                        inning_cover_buff = [inning_level, deal_card, called_list, own_pos, hold_cards, inning_major, agent_output]
                    
                    #更新对局缓存信息
                    level_card = env.Pokers2Num([s + '5' for s in __SUITSET__],[])
                    level_card.extend([c+54 for c in level_card])
                    mat_level_card = cards2matrix(level_card, inning_level, inning_major)
                    mat_score_card = cards2matrix([], inning_level, inning_major)
                    mat_remain_score_card = [s + '5' for s in __SUITSET__] + [s + '0' for s in __SUITSET__] + [s + 'K' for s in __SUITSET__]
                    mat_remain_score_card = env.Pokers2Num(mat_remain_score_card,[])
                    mat_remain_score_card.extend([c+54 for c in mat_remain_score_card])
                    mat_remain_score_card = cards2matrix(mat_remain_score_card, inning_level, inning_major)
                    mat_played_card = cards2matrix([], inning_level, inning_major)

                    round_play_seat = [0,0,0]
                    round_play_card = [0,0,0]
                    round_play_team = [0,0,0]

                    def_v = cards2matrix([], inning_level, inning_major)
                    history_play_card = deque([def_v]*15, maxlen=15)
                    def_v = np.zeros((__PLAYER_COUNT__,__PLAYER_COUNT__))
                    history_play_seat = deque([def_v]*15, maxlen=15)
                    def_v = np.zeros((__PLAYER_COUNT__,__PLAYER_COUNT__))
                    history_play_team = deque([def_v]*15, maxlen=15)

                #出牌阶段
                elif stage == "play":
                    history = env.getPlayHistory()
                    seat = env.getPlayerPosition()
                    hand_cards = env.getPlayerHoldCards(seat)
                    level = env.getLevel()
                    score = env.getTotalScore()
                    remain_score = 200-score
                    banker_seat = env.getBanker()
                    
                    fixed_action_card, discard_action_card, discard_num = env.getLegalActions(history[1], env.getPlayerHoldCards(seat) , level)
                    
                    obs_x = {}
                    obs_x['history_play_card'] = np.array(history_play_card)#历史出牌
                    obs_x['history_play_seat'] = np.array(history_play_seat)#出牌座位号
                    obs_x['history_play_team'] = np.array(history_play_team)#出牌阵营
                    obs_x['seat'] = seat
                    obs_x['hand_cards'] = hand_cards
                    obs_x['round_play_card'] = round_play_card
                    obs_x['round_play_seat'] = round_play_seat
                    obs_x['round_play_team'] = round_play_team
                    obs_x['played_cards'] = np.copy(mat_played_card)
                    obs_x['level_card'] = np.copy(mat_level_card)
                    obs_x['score_card'] = np.copy(mat_score_card)
                    obs_x['remain_score_card'] = np.copy(mat_remain_score_card)
                    obs_x['banker'] = banker_seat==seat or banker_seat==(seat+2)%4
                    mat_action_card = actor.playCard(obs_x, fixed_action_card, discard_action_card, discard_num, obs_x['banker'])
                    action = matrix2card(mat_action_card, level)
                    
                    #历史出牌分3部分，出牌，出牌座位号，阵营
                    response = [env.getPlayerPosition(), action]

                    #记录经验回放
                    input_x[seat] = obs_x

                    #更新回合缓存
                    fseat = env.getCurrRoundPlaySeat()
                    rounds  = len(env.getCurrRoundPlayHistory())
                    round_play_card[rounds] = cards2matrix(action,inning_level,inning_major)
                    round_play_seat[rounds] = seat
                    round_play_team[rounds] = (seat == fseat or (seat+2)%4 == fseat) and 1 or 2
                    

                #一回合结束
                elif stage == 'roundend':
                    

                    
                    

                    #更新出牌历史，包括出的牌、出牌玩家位置
                    last_round_play = env.getLastRoundPlayHistory()
                    last_round_seat = env.getLastRoundPlaySeat()
                    round_play_his = np.array((__PLAYER_COUNT__,2*4*14))
                    round_seat_his = np.array((__PLAYER_COUNT__,__PLAYER_COUNT__))
                    round_team_his = np.array((__PLAYER_COUNT__,__PLAYER_COUNT__))
                    for i, playcard in enumerate(last_round_play):
                        play_card_mat = cards2matrix(playcard,inning_level,inning_major)
                        round_play_his[i,:] = play_card_mat
                        mat_played_card += play_card_mat
                    history_play_card.append(round_play_his)
                    
                    for i in range(4):
                        round_seat_his[i,:] = get_one_hot_array((last_round_seat+i)%__PLAYER_COUNT__, __PLAYER_COUNT__)
                    history_play_seat.append(round_seat_his)

                    #出牌阵营，奖励
                    if last_round_seat==0 or last_round_seat==2:
                        round_team_his[0,:]=np.array([1,2,1,2])
                        round_team_his[2,:]=np.array([1,2,1,2])
                        round_team_his[1,:]=np.array([2,1,2,1])
                        round_team_his[3,:]=np.array([2,1,2,1])
                        reward = score == 0 and [1, -1, 1, -1] or [-score/10, score/10, -score/10, score/10]
                    else:
                        round_team_his[1,:]=np.array([1,2,1,2])
                        round_team_his[3,:]=np.array([1,2,1,2])
                        round_team_his[0,:]=np.array([2,1,2,1])
                        round_team_his[2,:]=np.array([2,1,2,1])
                        reward = score == 0 and [-1, 1, -1, 1] or [score/10, -score/10, score/10, -score/10]
                    history_play_team.append(round_team_his)

                    #更新捡到的分牌
                    score_pok = env.getLastRoundScorePoke()
                    mat_s = cards2matrix(score_pok, inning_level, inning_major)
                    mat_score_card += mat_s
                    mat_remain_score_card -= mat_s

                    #重置当前回合信息
                    round_play_seat = [0,0,0]
                    round_play_card = [0,0,0]
                    round_play_team = [0,0,0]
                    response = None
                    
                    
                    #存储经验回放
                    history_play_card_buff.append(np.array(history_play_card))
                    history_play_seat_buff.append(np.array(history_play_seat))
                    history_play_team_buff.append(np.array(history_play_team))
                    played_card_buff.append(mat_played_card)
                    level_card_buff.append(mat_level_card)
                    score_card_buff.append(mat_score_card)
                    remain_score_card_buff.append(mat_remain_score_card)
                    round_play_card_buff.append(round_play_his)
                    round_play_seat_buff.append(round_seat_his)
                    round_play_team_buff.append(round_team_his)
                    # banker_buff.append(mat_banker)

                #结束
                elif stage == 'finish':
                    #极牌转化为矩阵
                    lv = env.getLevel()
                    level_card = env.Pokers2Num([s + lv for s in __SUITSET__],[])
                    level_card.extend([c+54 for c in level_card])
                    mat_level_card = cards2matrix(level_card,inning_level,inning_major)
                    response = None
                    


                    
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                size[position] += 1
                position, obs, env_output = env.step(action)
                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)

                            episode_return = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
                            episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break

            for p in positions:
                while size[p] > T: 
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e