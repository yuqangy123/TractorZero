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

from games.tractors.models.bid_model import BidModel
from games.tractors.models.cover_model import CoverModel
from games.tractors.models.play_model import PlayModel

model_dict = {
    "banker": PlayModel,
    "banker_down": PlayModel,
    'banker_up': PlayModel,
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
        self._device = device
        
        self.models = {}
        for model_name, model_class in model_dict.items():
            self.models[model_name] = model_class().to(torch.device(device))
    
    def forward(self, position, z, x, legal_actions=None, training=False, flags=None, debug=False, game_infoset=None):
        model = self.models[position]
        result = model.forward(z, x, legal_actions, training, flags)        
        return result 
        
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

