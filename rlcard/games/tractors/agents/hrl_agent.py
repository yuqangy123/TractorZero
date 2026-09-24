from .agent import Agent
from ..model.models import Model
import torch, os

class HRLAgent(Agent):

    def __init__(self, device=None):
        self._model = Model(device=device)
        # 评估必须与训练 actor 一致用 eval 模式：网络含 BatchNorm
        self._model.eval()
        
    def loadModel(self, position, model_path, device):
        if not os.path.exists(model_path):
            print(f'HRLAgent: loadModel failed, position:{position}, model_patth:{model_path}')
            return
        self._model.get_model(position).load_state_dict(
            torch.load(model_path, map_location=device))
        
        
    def play(self, position, z_strategy, x_strategy, z, x, legal_actions, legal_card_type, infosets, flags=None):
        # z_strategy/x_strategy 为状态级表示（策略头输入），z/x 为按合法牌型批处理的表示（牌型/动作头输入）
        # 评估阶段不向模型透传 flags，避免 exp_epsilon 探索污染评估结果
        return self._model.play(position, z_strategy, x_strategy, z, x, legal_actions, legal_card_type)
    def cover(self, z, x, infosets, flags=None):
        return self._model.cover(z, x)
    def bid(self,  z, x, infosets, flags=None):
        return self._model.bid(z, x)
