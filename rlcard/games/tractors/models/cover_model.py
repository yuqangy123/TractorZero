import torch.nn as nn
import torch
import numpy as np
#埋牌模型

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=2, kernel_size=3, stride=1, padding=0):
        super(ResNet, self).__init__()
        self.in_channels = 14

        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=padding)

        # 残差层
        self.layer1 = self._make_layer(block, 14, layers[0])
        self.layer2 = self._make_layer(block, 28, layers[1], stride=stride)
        self.layer3 = self._make_layer(block, 56, layers[2], stride=stride)
        self.layer4 = self._make_layer(block, 112, layers[3], stride=stride)

        # 全局平均池化和全连接层
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(112, 1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x
    


class CoverModel(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        2*14*4扑克牌矩阵， 加上1*4矩阵代表花色，和1*14代表点数，
        
        当该局结束后，
        将n张牌中的非的回归值以输赢分，
        来作为target优化方向进行梯度更新
        '''
        self.resnet_hand = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=2)
        self.resnet_partner_bid = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=2)
        self.resnet_rival_bid = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=2)
        self.merged = nn.Conv2d(128,  128, 1)#特征融合
        
        
        
        layers = []
        for _ in range(3):
            layers.append(nn.Linear(128, 512))
            layers.append(nn.LeakyReLU())
        self.hid_layer = nn.Sequential(*layers)
        self.fc2 = nn.Linear(512, 1)


    def forward(self, own_cards, partner_bid_cards, rival_bid_cards, mask, return_val=False, exp_epsilon=None):
        own_x = self.resnet_hand(own_cards)
        partner_bid_x = self.resnet_partner_bid(partner_bid_cards)
        rival_bid_x = self.resnet_rival_bid(rival_bid_cards)
        
        x = own_x + partner_bid_x + rival_bid_x
        x = self.merged(x)

        x = torch.mm(x, mask)
        
        x = self.hid_layer(x)
        x = self.fc2(x)

        if return_val:
            return x
        else:
            if exp_epsilon is not None and exp_epsilon > 0 and np.random.rand() < exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[:8]
            return action
        pass

    def load_checkpoint(self, dict):
        pass