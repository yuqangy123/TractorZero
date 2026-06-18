import torch.nn as nn
import torch
import numpy as np
from rlcard.games.tractors.models.BasicBlockM import ResNet, ResidualBlock
from torch.distributions import Categorical

#埋牌模型

class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        '''
        2*4*14扑克牌矩阵， 加上1*4矩阵代表花色，和1*14代表点数，
        
        当该局结束后，
        将n张牌中的非的回归值以输赢分，
        来作为target优化方向进行梯度更新
        '''
        # 手牌特征提取器 (2,4,14) -> 256维
        #手牌 友方叫牌 对方叫牌 共用一个特征提取器
        self.cards_restnet = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=3)#kernel_size是3还是2要斟酌一下


        # self.resnet_hand = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=2)
        # self.resnet_partner_bid = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=2)
        # self.resnet_rival_bid = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=2)
        self.merged = nn.Conv2d(128,  128, 1)#特征融合
        
        
        
        layers = []
        for _ in range(3):
            layers.append(nn.Linear(128, 512))
            layers.append(nn.LeakyReLU())
        self.hid_layer = nn.Sequential(*layers)
        self.fc2 = nn.Linear(512, 1)
        self.sigmod = nn.Sigmoid()

    # 输出是0-1之间的值
    def forward(self, own_cards, partner_bid_cards, rival_bid_cards, level_cards, exp_epsilon=None):
        own_x = self.cards_restnet(own_cards)
        partner_bid_x = self.cards_restnet(partner_bid_cards)
        rival_bid_x = self.cards_restnet(rival_bid_cards)
        level_x = self.cards_restnet(level_cards)
        
        x = own_x + partner_bid_x + rival_bid_x + level_x
        x = self.merged(x)

        # x = torch.mm(x, mask)
        
        x = self.hid_layer(x)
        x = self.fc2(x)
        
        if exp_epsilon is not None and exp_epsilon > 0 and np.random.rand() < exp_epsilon:
            ret = torch.rand(x.shape[0], (1,))[0]
        else:
            ret = self.sigmod(x)
        return ret

    def load_checkpoint(self, dict):
        pass

class Critic(Actor):# 结构同Actor
    def __init__(self) -> None:
        super().__init__()
        self.sigmod = None

    def forward(self, own_cards, partner_bid_cards, rival_bid_cards, level_cards):
        own_x = self.cards_restnet(own_cards)
        partner_bid_x = self.cards_restnet(partner_bid_cards)
        rival_bid_x = self.cards_restnet(rival_bid_cards)
        level_x = self.cards_restnet(level_cards)
        
        x = own_x + partner_bid_x + rival_bid_x + level_x
        x = self.merged(x)

        # x = torch.mm(x, mask)
        
        x = self.hid_layer(x)
        ret = self.fc2(x)

        return ret
    

class CoverModel(nn.Module):
    def __init__(self):
        super().__init__()
        #手牌信息（2*4*14） 历史叫牌信息（3*（2*4*14））+ 历史叫牌座位（3*（4）） 级牌（2*4*14） 
        self.actor = Actor()
        self.critic = Critic()


    def evaluate(self, own_cards, partner_bid_cards, rival_bid_cards, level_cards, action_prob, exp_epsilon=None):
        
        action_probs = self.actor.forward(own_cards, partner_bid_cards, rival_bid_cards, level_cards, exp_epsilon)
        distribution = Categorical(action_probs)

        #作用：衡量一个动作在当前策略（Actor 输出的动作概率分布）下的可能性。
        #在 PPO 算法中，它被用来计算新旧策略之间的比率（ratio），以控制策略更新的幅度（通过 clip 机制）。
        action_log_prob = distribution.log_prob(action_prob)
        #计算分类分布的熵（衡量一个概率分布的不确定性或随机性）。熵越高 → 分布越“均匀” → 动作选择越随机（鼓励探索）。熵越低 → 分布越“集中” → 动作趋于确定性（利用已有策略）
        entropy = distribution.entropy()
        state_values = self.critic.forward(own_cards, partner_bid_cards, rival_bid_cards, level_cards)

        return action_log_prob, state_values, entropy

    def load_checkpoint(self, dict):
        pass