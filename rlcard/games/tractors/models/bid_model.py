import torch.nn as nn
import torch
import numpy as np
from rlcard.games.tractors.models.BasicBlockM import ResNet, ResidualBlock

class BidModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=3)
        # self.lstm = nn.LSTM(162, 128, batch_first=True)
        
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, 128)
        self.dense_score = nn.Linear(128, 16)
        self.dense_suit  = nn.Linear(128, 4)

    def forward(self, obs_x):
        x = self.resnet(obs_x)

        # bid_his = self.resnet(bid_his)
        # bid_his = torch.cat((bid_his, lstm_out[:, -1, :]), dim=1)
        # lstm_his_out, (h_n, _) = self.lstm(bid_his)
        # x = torch.cat((x, bid_card, lstm_his_out[:, -1, :]), dim=1)
        x = nn.LeakyReLU(self.dense1(x))
        x = nn.LeakyReLU(self.dense2(x))
        x = nn.LeakyReLU(self.dense3(x))
        x = nn.LeakyReLU(self.dense4(x))
        y_score = self.dense_score(x)
        y_suit = self.dense_suit(x)
        return y_score, y_suit

    def load_checkpoint(self, dict):
        pass