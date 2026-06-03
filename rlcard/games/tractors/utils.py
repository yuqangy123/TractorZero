import os 
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time

import torch 


###############################################################
# 牌面表示：数字
# h:红桃 d:方片 s:黑桃 c:草花 
# (0-h1 1-d1 2-s1 3-c1) (4-h2 5-d2 6-s2 7-c2) ... 52-joker 53-Joker (54-h1 55-d1 56-s1 57-c1) ... 106-joker 107-Joker
# 请注意：10记为0
# 共2副108张
###############################################################
__CARDSCALE__ = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
__SUITSET__ = ['s','h','c','d']# h:红桃 d:方片 s:黑桃 c:草花 
__MAJOR__ = ['jo', 'Jo']#小王 大王
__POINT__ = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']
__PLAYER_COUNT__ = 3
__CARDS_NUM__ = (108)

__HAND_CARD_NUM__ = 25#手牌数量

__MAX_SCORE__ = 40#最多分数，1个代表5分

#牌类型
__SINGLE__ = 0
__PAIR__ = 1
__TRACTOR__ = 2
__SUSPECT__ = 3

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 15: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]



def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers

def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = ['landlord', 'landlord_up', 'landlord_down']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 319 if position == 'landlord' else 430
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 54), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()

        while True:
            while True:
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

# level=11
# major=3
# matrix = np.arange(2*15*4, dtype=np.int8)
# matrix = matrix.reshape(2, 4, 15)
# matrix_cp = matrix.copy()
# print(matrix)
# view = matrix[:,:,level:-1]
# view[:]=np.roll(view, shift=-1, axis=2)
# print(matrix)
# matrix[:, [0,major], 0:13] = matrix[:, [major,0], 0:13]
# print(matrix)
# view = matrix[:,:,level:-1]
# view[:]=np.roll(view, shift=1, axis=2)
# print(matrix)
# matrix[:, [0,major], 0:13] = matrix[:, [major,0], 0:13]
# print(matrix)
# print(matrix == matrix_cp)
#扑克牌(number)转矩阵
# def cards2matrix(list_cards, level='K', major='s'):
def cards2matrix(list_cards):
    """
        'A','2','3','4','5','6','7','8','9','0','J','Q','K','o','O'
    s    0   0   0   0   0   0   0   0   0   0   0   0   0   1   1
    h    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    c    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    d    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    [2, 4, 15]
    kernel为2*2 更关注对子, 游戏中没有3和4


           s   h   c   d
    'A'    0   0   0   0
    '2'    0   0   0   0
    '3'    0   0   0   0
    '4'    0   0   0   0
    '5'    0   0   0   0
    '6'    0   0   0   0
    '7'    0   0   0   0
    '8'    0   0   0   0
    '9'    0   0   0   0
    '0'    0   0   0   0
    'J'    0   0   0   0
    'Q'    0   0   0   0
    'K'    0   0   0   0
    'o'    1   0   0   0
    'O'    1   0   0   0
    [2, 15, 4] 
    """
    matrix = np.zeros(54*2, dtype=np.int8)
    matrix[list_cards] = 1
    matrix = np.insert(matrix, 53, [0,0,0])
    matrix = np.insert(matrix, 57, [0,0,0])
    matrix = np.insert(matrix, 60+53, [0,0,0])
    matrix = np.insert(matrix, 60+57, [0,0,0])
    matrix = matrix.reshape(2,15,4)
    matrix = np.transpose(matrix, (0,2,1))

    # # 根据级数调整列顺序数组
    # # new_order = list(range(matrix.shape[2]))
    # level = __CARDSCALE__.index(level)
    # if level != 12:
    #     view = matrix[:,:,level:-2]
    #     view[:]=np.roll(view, shift=-1, axis=2)

    # # 根据主花色调整行序列数组
    # major = __SUITSET__.index(major)
    # if major != 0:
    #     matrix[:, [0,major], 0:13] = matrix[:, [major,0], 0:13]
    return matrix

#修改了，没做过测试
def matrix2cards(mat):    
    mat = np.transpose(mat, (0,2,1))
    cards = mat.flatten()
    # 获取指定索引的元素
    selected_cards = list(cards[0:52]) + [cards[52], cards[56]] + list(cards[60+0:60+52]) + [cards[60+52], cards[60+56]]
    # 获取selected_cards中值为1的索引
    indices = np.where(np.array(selected_cards) == 1)[0]
    return indices

# def matrix2cards(mat, level='K', major='s'):
#     level = __CARDSCALE__.index(level)
#     if level != 12:
#         view = mat[:,:,level:-2]
#         view[:]=np.roll(view, shift=1, axis=2)

#     major = __SUITSET__.index(major)
#     if major != 0:
#         mat[:, [0,major], 0:13] = mat[:, [major,0], 0:13]

#     mat = np.transpose(mat, (0,2,1))
#     cards = mat.flatten()
#     # 获取指定索引的元素
#     selected_cards = list(cards[0:52]) + [cards[52], cards[56]] + list(cards[60+0:60+52]) + [cards[60+52], cards[60+56]]
#     # 获取selected_cards中值为1的索引
#     indices = np.where(np.array(selected_cards) == 1)[0]
#     return indices

# card2mst = cards2matrix([66, 3, 56, 76, 40, 33, 48, 80, 52,53, 107,5], '2', 'c')
# mst2card = matrix2cards(card2mst,  '2', 'c')
# print(mst2card)
def get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    if num_left_cards>0 and num_left_cards<=max_num_cards:
        one_hot[num_left_cards - 1] = 1

    return one_hot

def get_full_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    if num_left_cards>0 and num_left_cards<=max_num_cards:
        one_hot[:num_left_cards] = 1

    return one_hot



def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _process_action_seq(sequence, length=15):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix


if __name__ == '__main__':
    # (0-h1 1-d1 2-s1 3-c1) (4-h2 5-d2 6-s2 7-c2) ... 52-joker 53-Joker (54-h1 55-d1 56-s1 57-c1) ... 106-joker 107-Joker
    list_cards = [0,4,16,17,18,19,52,53]
    for c in [0,4,16,17,18,19,52,53]:list_cards.append(c+54)        
    cards2matrix(list_cards)



__all__ = [
    '__CARDSCALE__',
    '__SUITSET__',
    '__MAJOR__',
    '__POINT__',
    "__PLAYER_COUNT__",
    "__CARDS_NUM__",
    '__MAX_SCORE__',
    "__HAND_CARD_NUM__",
    "cards2matrix",
    "get_one_hot_array",
    "get_full_hot_array",
    "matrix2cards",
    '__SINGLE__',
    '__PAIR__',
    '__TRACTOR__' ,
    '__SUSPECT__',
]