# Copyright 2021 RLCard Team of Texas A&M University
# Copyright 2021 DouZero Team of Kwai
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
from torch import nn
from rlcard.tractors.models.cover_model import CoverModel as CoverModel
from rlcard.tractors.models.bid_model import BidModel as BidModel


class TractorsModel:
    def __init__(
        self,
        exp_epsilon=0.01,
        device='cpu'
    ):
        self.__models = dict()
        self.__models['cover'] = CoverModel().to(device)
        self.__models['bid'] = BidModel().to(device)

    def share_memory(self):
        for agent in self.__models:
            agent.share_memory()

    def eval(self):
        for agent in self.__models:
            agent.eval()

    def parameters(self, index):
        return self.__models[index].parameters()

    def get_cover_model(self):
        return self.__models['cover']

    def get_bid_model(self):
        return self.__models['bid']
    
    def bidding(self, obs_x, bid_cards, left_num):
        return self.__models['bid'](obs_x, bid_cards, left_num)
    
    def cover(self, own_cards, partner_bid_cards, rival_bid_cards, mask):
        return self.__models['cover'](own_cards, partner_bid_cards, rival_bid_cards, mask)
    
    
