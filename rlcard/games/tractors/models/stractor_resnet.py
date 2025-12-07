import torch.nn as nn
import torch
import numpy as np
import math

'''
手牌特征提取网络Resnet

[b, 2, 14, 4] 是手牌矩阵
示例：
self.resnet_my_card = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=3)
'''




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        ##kernelsize=3, padding=1, stride=1以保存卷积后的尺寸不变化
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
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
    def __init__(self, block, layers=[2,2,2,2], in_channels=2, out_dim=256, hidden_channels=[14,28,56,112], kernel_size=3, stride=1, padding=0):
        super(ResNet, self).__init__()
        self.hidden_channels = hidden_channels[0]

        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.relu = nn.LeakyReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=padding)

        # 残差层
        self.reslayers = []
        for i in range(min(len(layers), len(hidden_channels))):
            self.reslayers.append(self._make_layer(block, hidden_channels[i], layers[i], stride=stride))

        # 全局平均池化和全连接层
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #融合卷积层
        self.conv_fusion = nn.Linear(hidden_channels[len(hidden_channels)-1]*4*15, out_dim)#nn.Conv2d(hidden_channels[len(hidden_channels)-1], out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self._init_lstm_weights()

    def toDevice(self, device):
        for i in range(len(self.reslayers)):
            self.reslayers[i].to(device)

    def _init_lstm_weights(self):
        """初始化LSTM权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1，有助于记忆长期依赖
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.hidden_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.hidden_channels, out_channels, stride, downsample))
        self.hidden_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_dim = len(x.shape)
        if x_dim == 5:
            #将回合维度数据合并到bath批次维度
            batch_size = x.size(0)
            seq_len = x.size(1)
             # 重塑为 [batch_size*seq_len, 2, 4, 15]
            x = x.view(-1, x.size(2), x.size(3), x.size(4))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        for i in range(len(self.reslayers)):
            x = self.reslayers[i](x)
            
        # 展平特征图
        x = torch.flatten(x, 1)
        # 应用融合层
        x = self.conv_fusion(x)
        
        if x_dim == 5:
            # 重塑回批次格式 [batch_size, seq_len, out_dim]
            x = x.view(batch_size, seq_len, -1)
        return x
    