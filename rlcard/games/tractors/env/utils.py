import os 
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time

import torch 


'''###############################################################
牌面表示：数字
h:红桃 d:方片 s:黑桃 c:草花 
(0-h1 1-d1 2-s1 3-c1) (4-h2 5-d2 6-s2 7-c2) ... 52-joker 53-Joker (54-h1 55-d1 56-s1 57-c1) ... 106-joker 107-Joker
0 1 2 3         # A
4 5 6 7         # 2
8 9 10 11       # 3
12 13 14 15     # 4
16 17 18 19     # 5
20 21 22 23     # 6
24 25 26 27     # 7
28 29 30 31     # 8
32 33 34 35     # 9
36 37 38 39     # 10
40 41 42 43     # J 
44 45 46 47     # Q 
48 49 50 51     # K
52 53           # joker


# 请注意：10记为0
# 共2副108张
###############################################################'''
__CARDSCALE__ = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
__SUITSET__ = ['s','h','c','d']# h:红桃 d:方片 s:黑桃 c:草花 
__MAJOR__ = ['jo', 'Jo']#小王 大王
__POINT__ = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']
__PLAYER_COUNT__ = 3
__CARDS_NUM__ = (108)

__HAND_CARD_NUM__ = 28#手牌数量

__MAX_SCORE__ = 40#最多分数，1个代表5分

#牌类型
__SINGLE__ = 0
__PAIR__ = 1
__TRACTOR__ = 2
__SUSPECT__ = 3
__DISCARD__ = 4
__WRONG__ = 5

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
def cards2matrix(list_cards, major='s', level='2'):
    """
        '3','4','5','6','7','8','9','10','J','Q','K','A','2','o','O'
    s    0   0   0   0   0   0   0   0   0   0   0   0   0   1   1
    h    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    c    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    d    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    [2, 4, 15]
    kernel为2*2 更关注对子, 游戏中没有3和4


           s   h   c   d
    '3'    0   0   0   0
    '4'    0   0   0   0
    '5'    0   0   0   0
    '6'    0   0   0   0
    '7'    0   0   0   0
    '8'    0   0   0   0
    '9'    0   0   0   0
    '10'   0   0   0   0
    'J'    0   0   0   0
    'Q'    0   0   0   0
    'K'    0   0   0   0
    'A'    0   0   0   0
    '2'    0   0   0   0 52
    'o'    1   0   0   0 56
    'O'    1   0   0   0 60
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
    
    #环境的牌值是从A-K，o，O，将A和2放o前面，方便卷积提取牌型特征，
    new_order = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 13, 14]
    
    # # 根据级数调整列顺序数组
    # # new_order = list(range(matrix.shape[2]))
    level = __CARDSCALE__.index(level)
    if level != 1:
        if level == 0:
            new_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14]
        else:
            new_order = list(range(1, 15))
            new_order.insert(-2, 0)  # 在12后面插入
            del new_order[level-1]
            new_order.insert(-2, level)  # 插入级牌
            
        # view = matrix[:,:,level:-2]
        # view[:]=np.roll(view, shift=-1, axis=2)
    matrix = matrix[:, :, new_order]

    # # 根据主花色调整行序列数组
    major = __SUITSET__.index(major)
    if major != 0:
        matrix[:, [0,major], 0:13] = matrix[:, [major,0], 0:13]
    return matrix


def matrix2cards(matrix, major='s', level='2'):
    #转换花色
    major = __SUITSET__.index(major)
    if major != 0:
        matrix[:, [0,major], 0:13] = matrix[:, [major,0], 0:13]
    
    #转换级牌
    new_order = [11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]
    level = __CARDSCALE__.index(level)
    if level != 1:
        if level == 0:
            new_order = [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
        else:
            new_order = [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]
            new_order.insert(level, 12)
        # view = mat[:,:,level:-2]
        # view[:]=np.roll(view, shift=1, axis=2)
    matrix = matrix[:, :, new_order]
    

    matrix = np.transpose(matrix, (0,2,1))
    cards = matrix.flatten()
    # 获取指定索引的元素
    selected_cards = list(cards[0:52]) + [cards[52], cards[56]] + list(cards[60+0:60+52]) + [cards[60+52], cards[60+56]]
    # 获取selected_cards中值为1的索引
    indices = np.where(np.array(selected_cards) == 1)[0]
    return indices

def test_matrix_cards_transform():
    def __num2Poker__(num): # num: int-[0,107]
            # Already a poker
            # if type(num) is str and (num in self.Major or (num[0] in __SUITSET__ and num[1] in __CARDSCALE__)):
            #     return num
            # Locate in 1 single deck
            NumInDeck = num % 54
            # joker and Joker:
            if NumInDeck == 52:
                return "jo"
            if NumInDeck == 53:
                return "Jo"
            # Normal cards:
            pokernumber = __CARDSCALE__[NumInDeck // 4]
            pokersuit = __SUITSET__[NumInDeck % 4]
            return pokersuit + pokernumber
        
    poker_test = [3, 5, 33, 40, 48, 52, 53, 56, 66, 76, 80, 107]
    print([__num2Poker__(p)for p in poker_test])
    poker_test = np.array(poker_test)

    for major in ['s', 'h', 'c', 'd']:
        for level in ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']:
            card2mst = cards2matrix(poker_test, major, level)
            mst2card_test = matrix2cards(card2mst, major, level)
            assert (mst2card_test == poker_test).all(), f"Failed for major={major}, level={level}"
# test_matrix_cards_transform()
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
    '__DISCARD__',
    '__WRONG__',
    '__HAND_CARD_NUM__',
]