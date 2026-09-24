import torch.nn as nn
import torch as t
import numpy as np
import torch.nn.functional as F
from .BasicBlockM import ResNet, ResidualBlock
from ..env.utils import __BID_ACTION_NUM__

class BidModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 1024
        
        self.resnet = ResNet(ResidualBlock, layers = [2,2,2,2 ], hidden_channels=[14,28,56,112], \
                                        in_channels=10,out_dim=hidden_dim, kernel_size=3, padding=1, stride=1)
        # self.lstm = nn.LSTM(162, 128, batch_first=True)
        
        # obs_z dim
        z_dim = 20
        self.dense1 = nn.Linear(hidden_dim + z_dim, 2048)
        self.dense2 = nn.Linear(2048, 1024)
        self.dense3 = nn.Linear(1024, 512)
        self.dense4 = nn.Linear(512, 256)
        self.dense5 = nn.Linear(256, 128)
        self.dense_action = nn.Linear(128, 1)

    def forward(self, obs_z, obs_x, return_value=False, flags=None):
        x = self.resnet(obs_x)
        
        x = t.cat([obs_z, x], dim=1)
        x = F.leaky_relu_(self.dense1(x))
        x = F.leaky_relu_(self.dense2(x))
        x = F.leaky_relu_(self.dense3(x))
        x = F.leaky_relu_(self.dense4(x))
        x = F.leaky_relu_(self.dense5(x))
        output = self.dense_action(x)
    
        #随机探索
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            out_action = t.randint(output.shape[0], (1,))[0]
            
        else:
            out_action = t.argmax(output, dim=0)[0]
            
        if return_value:   
            return dict(action=out_action, values=(output,))
        else:
            return dict(action=out_action)
        
        

    def load_checkpoint(self, dict):
        pass
    
    
    
    