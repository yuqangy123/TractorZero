import torch.nn as nn
import torch as t
import numpy as np
from rlcard.games.tractors.models.BasicBlockM import ResNet, ResidualBlock

class BidModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 512
        
        self.cards_encoder = ResNet(ResidualBlock, layers = [2,2,2,2 ], hidden_channels=[14,28,56,112], \
                                        in_channels=2,out_dim=hidden_dim, kernel_size=3, padding=1, stride=1)
        # self.lstm = nn.LSTM(162, 128, batch_first=True)
        
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, 128)
        self.dense_score = nn.Linear(128, 16)
        self.dense_suit  = nn.Linear(128, 4)

    def forward(self, obs_z, obs_x, mask, return_value=False, flags=None):
        x = self.resnet(obs_x)
        
        x = t.cat([obs_z, obs_z, obs_z, obs_z, x], dim=1)
        x = nn.LeakyReLU(self.dense1(x))
        x = nn.LeakyReLU(self.dense2(x))
        x = nn.LeakyReLU(self.dense3(x))
        x = nn.LeakyReLU(self.dense4(x))
        y_score = self.dense_score(x)
        y_suit = self.dense_suit(x)
        y_score = y_score * mask
    
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            #随机探索
            out_score = t.multinomial(mask, num_samples=1).squeeze(1)
            out_suit = np.random.randint(0, 4)
            
        else:
            out_score = t.argmax(y_score, dim=1)[0]
            out_suit = t.argmax(y_suit, dim=1)[0]
        # return dict(score=y_score_out, y_suit=y_suit_out, score_values=y_score, suit_values=y_suit)
        if return_value:   
            return dict(score=out_score, suit=out_suit, values=(y_score, y_suit))
        else:
            return dict(score=out_score, suit=out_suit)
        
        

    def load_checkpoint(self, dict):
        pass
    
    
    
    