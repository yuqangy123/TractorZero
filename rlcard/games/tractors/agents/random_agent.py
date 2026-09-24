import random
import torch

from .agent import Agent


# 随机选择的Agent
class RandomAgent(Agent):
    """随机动作Agent，作为评估和训练的基线对手。"""

    def play(self, position, z_strategy, x_strategy, z, x, legal_actions, legal_card_type, infosets, flags=None):
        # 随机Agent不使用状态级表示 z_strategy/x_strategy
        # 随机选择一个合法牌型
        action_type_index = random.randrange(len(legal_card_type))
        action_type = legal_card_type[action_type_index]
        # 在该牌型下随机选择一个具体动作
        action = random.randrange(len(legal_actions[action_type]))
        return dict(action_type_index=action_type_index, action=action)

    def cover(self, z, x, infosets, flags=None):
        # 返回随机logits，由上层结合合法mask取出8张埋牌
        action = torch.rand(x.shape[0], 2 * 4 * 15)
        return dict(action=action)

    def bid(self, z, x, infosets, flags=None):
        # x的batch大小即合法叫主动作数（首项恒为“不叫”），随机选择其一
        action = torch.randint(x.shape[0], (1,))
        return dict(action=action)