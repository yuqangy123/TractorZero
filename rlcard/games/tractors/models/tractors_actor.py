import time
import pickle
import torch
from argparse import ArgumentParser
from multiprocessing import Process
from random import randint
# import sys
import psutil
import os
# import objgraph
import gc

import numpy as np
import zmq
from pathlib import Path
from tractors.models.ppo_model import MLPActorCritic, MLPQNetwork
from pyarrow import deserialize, serialize
from utils import logger
from rlcard.tractors.utils import *
from rlcard.tractors.models.tractors_model import TractorsModel as Model

from rlcard.games.tractors.models.bid_model import BidModel
from rlcard.games.tractors.models.cover_model import CoverModel
from rlcard.games.tractors.models.play_model import PlayModel


ActionNumber = 2
# torch.set_num_threads(8)
class TractorsActor():
    def __init__(self, device='cpu', args={}):
        self.models = {}
        self.models['bid'] = BidModel(args).to(device)
        self.models['cover'] = CoverModel(args).to(device)
        self.models['banker'] = PlayModel(args).to(device)
        self.models['player'] = PlayModel(args).to(device)
        


    def biddingMajor(self, get_card, hold_card, bid_card, own_seat, bid_history, level):
        '''
            owned 自己的位置
            called 首次报主的玩家位置
            snatched 反主的玩家位置
        '''
        hold_card = get_card+hold_card
        hold_card = cards2matrix(hold_card, level)
        left_num = 25 - sum(hold_card)
        left_num = get_one_hot_array(left_num)

        bid_card = cards2matrix(bid_card, level)

        #友方的牌
        partner_called = []
        for bid_pro in bid_history:
            if bid_pro[0] != own_seat and (bid_pro[0]+2)%__PLAYER_COUNT__ == own_seat:
                partner_called = bid_pro[1]
                break                
        partner_called = cards2matrix(partner_called, level)

        pre = self.models.bidding(hold_card, bid_card, left_num)
        return pre > 0.5

    def coverCard(self, public_card, hold_card, own_seat, bid_history, level, major):
        mask = hold_card

        hold_card = public_card+hold_card
        hold_card = cards2matrix(hold_card, level, major)

        partner_called, rival_called = [],[]
        for bid_pro in bid_history:
            if bid_pro[0] != own_seat:
                if (bid_pro[0]+2)%__PLAYER_COUNT__ == own_seat:
                    partner_called = bid_pro[1]
                else:
                    rival_called = bid_pro[1]
                
        hold_card = cards2matrix(hold_card, level, major)
        partner_called = cards2matrix(partner_called, level, major)
        rival_called = cards2matrix(rival_called, level, major)
        
        cover_card = self.models.cover(hold_card, partner_called, rival_called, mask)
        cover_card = matrix2card(cover_card)
        return cover_card

    def playCard():
        pass

    def load_state_dict(self, model_name, dict):
        if model_name in self.models:
            self.models[model_name].load_checkpoint(dict)
            return True
        return False


class PlayerModel():
    def __init__(self, args) -> None:
        # 数据初始化
        # self.mb_states, self.mb_legal_indexs, self.mb_rewards, self.mb_actions, self.mb_dones, self.mb_values, self.mb_neglogp = [], [], [], [], [], [], []
        self.mb_states, self.mb_state_cut, self.mb_rewards, self.mb_actions, self.mb_dones, self.mb_values, self.mb_neglogp, self.mb_legal_indexs = [], [], [], [], [], [], [], []
        # self.all_mb_states, self.all_mb_legal_indexs, self.all_mb_rewards, self.all_mb_actions, self.all_mb_dones, self.all_mb_values, self.all_mb_neglogp = [], [], [], [], [], [], []
        self.all_mb_states, self.all_mb_state_cut, self.all_mb_rewards, self.all_mb_actions, self.all_mb_dones, self.all_mb_values, self.all_mb_neglogp, self.all_mb_legal_indexs = [], [], [], [], [], [], [], []
        self.args = args
        self.step = 0
        self.send_times = 1
        self.wflag = 0
        self.step_record = [0] * 10

        # 模型初始化
        self.model_id = -1
        self.new_weights = None
        self.model = MLPActorCritic((ActionNumber, 516 + ActionNumber * 54), ActionNumber)
        self.model_q = MLPQNetwork(567)
        with open('/home/zhaoyp/guandan_tog/actor_torch/q_network.ckpt', 'rb') as f:  # load DMC model
            tf_weights = pickle.load(f)
        self.model_q.load_tf_weights(tf_weights)

        # 连接learner
        context = zmq.Context()
        context.linger = 0  # For removing linger behavior
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{self.args.ip}:{self.args.data_port}')

        # log文件
        self.args.exp_path += f'/Client{args.client_index}'
        create_experiment_dir(self.args, f'Client{args.client_index}-')
        # self.args.ckpt_path = self.args.exp_path / 'ckpt'
        self.args.log_path = self.args.exp_path / 'log'
        # self.args.ckpt_path.mkdir()
        self.args.log_path.mkdir()
        logger.configure(str(self.args.log_path))

        # 模型文件路径
        self.args.ckpt_path = Path('/home/zhaoyp/guandan_tog/learner_torch/ckpt_bak')
        # 开模型订阅
        # subscriber = Process(target=run_weights_subscriber, args=(self.args, None))
        # subscriber.start()

        # 初始化模型
        print('set weight start')
        model_init_flag = True
        while not model_init_flag:
            new_weights, self.model_id = find_new_weights(-1, self.args.ckpt_path)
            if new_weights is not None:
                self.model.load_state_dict(new_weights)
                model_init_flag = True
        print('set weight success')

    def sample(self, state) -> int:
        states = state['x_batch']
        state_no_action = state['x_no_action']
        legal_action = ActionNumber
        legal_index = np.ones(ActionNumber)
        if len(states) >= ActionNumber:
            indexs = self.model_q.get_max_n_index(states, ActionNumber)
            dqn_states = np.asarray(states[indexs])
            top_actions = dqn_states[:, -54:].flatten()
            states = np.concatenate((state_no_action, top_actions))
        elif len(states) < ActionNumber:
            legal_action = len(states)
            legal_index[legal_action:] = np.zeros(ActionNumber-legal_action)
            top_indexs = self.model_q.get_max_n_index(states, ActionNumber)
            dqn_states = np.asarray(states[top_indexs])
            top_actions = dqn_states[:, -54:].flatten()
            states = np.concatenate((state_no_action, top_actions))  # 把动作先添加进来
            supple = -1 * np.ones(54 * (ActionNumber - legal_action))
            states = np.concatenate((states, supple))
            indexs = list(range(ActionNumber))

        action, value, neglogp = self.model.step(states, legal_index)
        self.step += 1
        self.mb_states.append(states)
        self.mb_state_cut.append(state_no_action)
        self.mb_legal_indexs.append(legal_index)
        self.mb_actions.append(action)
        self.mb_values.append([value])
        self.mb_neglogp.append([neglogp])
        return indexs[action]
        
    def update_weight(self):
        new_weights, self.model_id = find_new_weights(self.model_id, self.args.ckpt_path)
        if new_weights is not None:
            self.model.load_state_dict(new_weights)
            self.new_weights = None

    def save_data(self, reward):
        self.mb_rewards = [[0] for _ in range(len(self.mb_states))]
        self.mb_dones = [[0] for _ in range(len(self.mb_states))]
        if len(self.mb_states) != 0:
            self.mb_rewards[-1] = [reward]
            self.mb_dones[-1] = [1]
        self.all_mb_states += self.mb_states
        self.all_mb_state_cut += self.mb_state_cut
        self.all_mb_legal_indexs += self.mb_legal_indexs
        self.all_mb_rewards += self.mb_rewards
        self.all_mb_actions += self.mb_actions
        self.all_mb_dones += self.mb_dones
        self.all_mb_values += self.mb_values
        self.all_mb_neglogp += self.mb_neglogp
        
        self.mb_states, self.mb_state_cut, self.mb_legal_indexs, self.mb_rewards, self.mb_actions, self.mb_dones, self.mb_values, self.mb_neglogp = [], [], [], [], [], [], [], []

    def send_data(self, reward):
        # 调整数据格式并发送
        self.prepare_training_data(reward)
        self.socket.send(self.data)
        self.socket.recv()
        self.data = None

        # 打印log
        if self.send_times % 10 == 0:
            self.send_times = 1
            logger.record_tabular(f"ep_step of {self.args.client_index}" ,np.mean(self.step_record))
            logger.dump_tabular()
        else:
            self.step_record[self.send_times%10] = self.step
            self.send_times += 1

        # 重置数据存储
        self.step = 0
        self.mb_states, self.mb_legal_indexs, self.mb_rewards, self.mb_actions, self.mb_dones, self.mb_values, self.mb_neglogp = [], [], [], [], [], [], []
        self.all_mb_states, self.mb_state_cut, self.all_mb_rewards, self.all_mb_actions, self.all_mb_dones, self.all_mb_values, self.all_mb_neglogp, self.all_mb_legal_indexs = [], [], [], [], [], [], [], []

    def prepare_training_data(self, reward):
        # Hyperparameters
        gamma = 0.99
        lam = 0.95

        self.states = np.asarray(self.all_mb_states)
        self.state_cut = np.asarray(self.all_mb_state_cut)
        self.legal_indexs = np.asarray(self.all_mb_legal_indexs)
        self.rewards = np.asarray(self.all_mb_rewards)
        self.actions = np.asarray(self.all_mb_actions)
        self.dones = np.asarray(self.all_mb_dones)
        self.values = np.asarray(self.all_mb_values)
        self.neglogps = np.asarray(self.all_mb_neglogp)
        
        self.mb_states, self.mb_state_cut, self.mb_legal_indexs, self.mb_rewards, self.mb_actions, self.mb_dones, self.mb_values, self.mb_neglogp = [], [], [], [], [], [], [], []
        self.all_mb_states, self.mb_state_cut, self.all_mb_rewards, self.all_mb_actions, self.all_mb_dones, self.all_mb_values, self.all_mb_neglogp, self.all_mb_legal_indexs = [], [], [], [], [], [], [], []

        if reward[0] == 'y':
            self.rewards[-1][0] += 1
        else:
            self.rewards[-1][0] -= 1

        self.values = np.concatenate([self.values, [[0.0]]])
        self.deltas = self.rewards + gamma * self.values[1:] * (1.0 - self.dones) - self.values[:-1]
        
        nsteps = len(self.states)
        self.advs = np.zeros_like(self.rewards, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(nsteps)):
            self.nextnonterminal = 1.0 - self.dones[t]
            self.advs[t] = lastgaelam = self.deltas[t] + gamma * lam * self.nextnonterminal * lastgaelam
            
        def sf01(arr):
            """
            swap and then flatten axes 0 and 1
            """
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        
        self.returns = self.advs + self.values[:-1]
        self.data = [self.states, self.state_cut, self.actions, self.legal_indexs] + [sf01(arr) for arr in [self.returns, self.advs, self.neglogps]]
        name = ['obs', 'obs_cut', 'act', 'legal', 'ret', 'adv', 'logp']
        self.data = serialize(dict(zip(name, self.data))).to_buffer()


def run_one_player(index, args):
    args.client_index = index
    player = Player(args)

    # 初始化zmq
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f'tcp://*:{6000+index}')
    except:
        os.system("bash /home/zhaoyp/guandan_tog/actor_torch/rekill.sh")
        exit()


    while True:
        # 做动作到获得reward
        state = deserialize(socket.recv())
        # print(state)
        if not isinstance(state, int) and not isinstance(state, float) and not isinstance(state, str):
            action_index = player.sample(state)
            socket.send(serialize(action_index).to_buffer())
        elif isinstance(state, str):
            socket.send(b'none')
            if state[0] == 'y':
                player.save_data(int(state[1]))
            else:
                player.save_data(-int(state[1]))
            player.send_data(state)
            rss = float(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)                    
            # print(rss, 'GB')
            if rss > 0.7:
                os.system("bash /home/zhaoyp/guandan_tog/actor_torch/rekill.sh")
                exit()
            player.update_weight()
        else:
            socket.send(b'none')
            player.save_data(state)


def main():
    # 参数传递
    args, _ = parser.parse_known_args()

    def exit_wrapper(index, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_player(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(players):
                    if _i != index:
                        _p.terminate()

    players = []
    for i in range(4):
        # print(f'start{i}')
        p = Process(target=exit_wrapper, args=(i, args))
        p.start()
        time.sleep(0.5)
        players.append(p)

    for player in players:
        player.join()


import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

if __name__ == '__main__':
    main()
