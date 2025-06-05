import torch.nn as nn
import torch
import numpy as np
from rlcard.games.tractors.models.stractor_resnet import ResNet, ResidualBlock

class BidModel(nn.Module):
    def __init__(self):
        super().__init__()
        # [b, 2, 14, 4] 是手牌矩阵
        self.resnet_my_card = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=3)
        self.resnet_bid_card = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=3)
        # self.lstm = nn.LSTM(162, 128, batch_first=True)
        
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 1)

    def forward(self, obs_x, bid_card, left_num):
        #提取牌面信息
        x = self.resnet_my_card(obs_x)
        bid_card = self.resnet_bid_card(bid_card)
        x = torch.cat([x, bid_card, left_num], dim=1)
        x = x.unsqueeze(1).repeat(3)

        # bid_his = self.resnet(bid_his)
        # bid_his = torch.cat((bid_his, lstm_out[:, -1, :]), dim=1)
        # lstm_his_out, (h_n, _) = self.lstm(bid_his)
        # x = torch.cat((x, bid_card, lstm_his_out[:, -1, :]), dim=1)
        x = nn.LeakyReLU(self.dense1(x))
        x = nn.LeakyReLU(self.dense2(x))
        x = nn.LeakyReLU(self.dense3(x))
        x = nn.LeakyReLU(self.dense4(x))
        y = self.dense5(x)
        return y

    def load_checkpoint(self, dict):
        pass