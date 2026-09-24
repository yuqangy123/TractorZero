"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""
import math
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
import os

from .bid_model import BidModel
from .cover_model import CoverModel
from .play_model import BankerModel, IdlerModel, NUM_STRATEGY
from ..env.utils import *

model_dict = {
    "banker": BankerModel,
    "banker_op": BankerModel,
    "banker_down": IdlerModel,
    'banker_up': IdlerModel,
    "bid": BidModel,
    "cover": CoverModel,
}


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self._device = torch.device(device)
        
        self.models = {
            "banker": BankerModel(__WRONG__, self._device).to(self._device),
            "banker_op": BankerModel(__WRONG__, self._device).to(self._device),
            "banker_down": IdlerModel(__WRONG__, self._device).to(self._device),
            'banker_up': IdlerModel(__WRONG__, self._device).to(self._device),
            "bid": BidModel().to(self._device),
            "cover": CoverModel().to(self._device),
        }
    
    def bid(self,  z, x, flags=None):
        model = self.models['bid']
        z = z.to(self._device)
        x = x.to(self._device)
        output = model.forward(z, x, flags=flags)
        return output
    
    def cover(self, z, x, flags=None):
        model = self.models['cover']
        z = z.to(self._device)
        x = x.to(self._device)
        output = model.forward(z, x, flags=flags)
        return output
        
    def play(self, position, z_strategy, x_strategy, z, x, legal_actions, legal_card_type, flags=None):
        model = self.models[position]

        # 状态级表示（不含牌型候选）：策略头训练与推理共用同一输入
        _tzs = z_strategy.unsqueeze(0).float().to(self._device)
        _txs = x_strategy.unsqueeze(0).float().to(self._device)
        _tz = z.to(self._device)
        _tx = x.to(self._device)
        

        # 上层策略导向：先做状态级策略决策，再以此为条件预测牌型/具体出牌
        strategy_output = model.forward_strategy(_tzs, _txs, flags=flags)
        strategy = strategy_output['action']
        
        strategy_embed = torch.zeros(NUM_STRATEGY).float().to(self._device)
        # strategy_embed[strategy] = 1.0
        strategy_embed_input = strategy_embed.unsqueeze(0).expand(_tz.shape[0], -1)

        # 先预测出牌牌型（受策略导向条件）
        tp_output = model.forward_tp(_tz, _tx, strategy_embed_input, flags=flags)
        action_type_index = tp_output['action']
        action_type = legal_card_type[action_type_index]
        
        if flags and hasattr(flags, '_game_log'):
            print(f"策略:{strategy}, 动作牌型:{action_type}")
            
        #拼接动作
        action_batch = torch.from_numpy(legal_actions[action_type]).to(self._device)
        num_legal_actions = len(action_batch)
        _az = _tz[action_type_index].clone()
        _ax = _tx[action_type_index].clone()
        
        # 分块前向：限制单次 batch 大小，避免甩牌等大量合法动作导致显存峰值过高
        chunk_size = 256
        num_chunks = (num_legal_actions + chunk_size - 1) // chunk_size
        out_all = []
        win_rate_logits = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_legal_actions)
            n = end - start

            z_chunk = _az.unsqueeze(0).expand(n, -1).to(self._device)
            x_chunk = _ax.unsqueeze(0).expand(n, -1, -1, -1)
            x_chunk = torch.cat([action_batch[start:end], x_chunk], dim=1).to(self._device)
            strategy_embed_input = strategy_embed.unsqueeze(0).expand(n, -1)
            
            out_chunk = model.forward_act(z_chunk, x_chunk, strategy_embed_input,
                                          return_value=True, flags=flags)
            win_rate, win, lose = out_chunk['values']
            win_rate_logits.append(win_rate.detach())  # 原始胜率 logit（sigmoid 前），用于塌缩诊断
            _win_rate = torch.sigmoid(win_rate)  # 训练回归 1{wp>0}（BCEWithLogits），输出是胜率 logit，sigmoid 才是概率
            out = _win_rate * win + (1. - _win_rate) * lose
            out_all.append(out)

        out_all = torch.cat(out_all, dim=0)
        # 胜率 logit 塌缩诊断：均值为先验胜率对应 logit，std 反映状态条件信息量
        win_rate_logits = torch.cat(win_rate_logits, dim=0).float()
        win_rate_logit_mean = float(win_rate_logits.mean().cpu())
        # unbiased=False + numel 保护：唯一合法动作时标准差定义为 0，避免 dof<=0 告警
        win_rate_logit_std = (float(win_rate_logits.std(unbiased=False).cpu())
                              if win_rate_logits.numel() > 1 else 0.0)

        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = torch.randint(out_all.shape[0], (1,))[0]
        else:
            action = torch.argmax(out_all, dim=0)[0]
        
        return dict(action_type_index=action_type_index, action=action, strategy=strategy_embed,
                    win_rate_logit_mean=win_rate_logit_mean,
                    win_rate_logit_std=win_rate_logit_std)
    
    
   
           
    def share_memory(self):
        for k,v in self.models.items():
            v.share_memory()

    def eval(self):
        for k,v in self.models.items():
            v.eval()
            
    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models

