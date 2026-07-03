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

from rlcard.games.tractors.models.bid_model import BidModel
from rlcard.games.tractors.models.cover_model import CoverModel
from rlcard.games.tractors.models.play_model import BankerModel, IdlerModel
from rlcard.games.tractors.env.utils import *

model_dict = {
    "banker": BankerModel,
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
            "banker": BankerModel(4, self._device).to(self._device),
            "banker_down": IdlerModel(4, self._device).to(self._device),
            'banker_up': IdlerModel(4, self._device).to(self._device),
            "bid": BidModel().to(self._device),
            "cover": CoverModel().to(self._device),
        }
    
    def forward(self, position, z, x, legal_actions, flags=None):
        model = self.models[position]
        
        if position in ['banker', 'banker_down', 'banker_up']:
            action_tp_mask = [len(actions) for actions in legal_actions]
            # 若有多个牌型可选，先预测出牌牌型
            if sum(action_tp_mask) > 1:
                tp_output = model.forward_tp(z, x, torch.tensor(action_tp_mask).to(self.device), flags=flags)
                action_type = tp_output['action']
            else:
                action_type = next(i for i, count in enumerate(action_tp_mask) if count > 0)
            
            action_tp_mask = torch.zeros(__WRONG__).to(self.device)
            action_tp_mask[action_type] = 1.0
            _z = torch.cat([z, action_tp_mask], dim=1)
            
            _x_batch = np.repeat(
                x[np.newaxis, :, :],
                len(legal_actions[action_type]), axis=0)
            x_batch = np.concatenate((legal_actions[action_type], _x_batch), axis=1)
            
            output = model.forward_act(_z, x_batch, flags=flags)['action']
            return output
        
        elif position == 'bid':
            output = model.forward(z, x, legal_actions, flags=flags)
            return output
            
        elif position == 'cover':
            output = model.forward(z, x, legal_actions, flags=flags)['action']
            return output
        
        else:
            #抛出异常
            raise Exception('Invalid position')
    
    def learn_action_tp(self, position, z, x, action_tp_mask):
        if position in ['banker', 'banker_down', 'banker_up']:
            model = self.models[position]
            output = model.forward_tp(z, x, torch.tensor(action_tp_mask).to(self.device), return_value=True)
            return output
           
    
    def learn_action_play(self, position, z, x, action_type, legal_actions):
        if position in ['banker', 'banker_down', 'banker_up']:
            model = self.models[position]
            action_tp_mask = torch.zeros(__WRONG__).to(self.device)
            action_tp_mask[action_type] = 1.0
            _z = torch.cat([z, action_tp_mask], dim=1)
            
            _x_batch = np.repeat(
                x[np.newaxis, :, :],
                len(legal_actions[action_type]), axis=0)
            x_batch = np.concatenate((legal_actions[action_type], _x_batch), axis=1)
            
            output = model.forward_act(_z, x_batch, return_value=True)
            return output
        
    def learn_bid(self, position, z, x, legal_actions):
        if position == 'bid':
            model = self.models[position]
            output = model.forward(z, x, legal_actions, return_value=True)
            return output
           
    def learn_cover(self, position, z, x, legal_actions):
        if position == 'cover':
            model = self.models[position]
            output = model.forward(z, x, legal_actions, return_value=True)
            return output
    
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

