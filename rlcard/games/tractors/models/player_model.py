import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
import numpy as np
import scipy.signal
import gym
import os
import datetime
from rlcard.games.tractors.models.stractor_resnet import ResNet, ResidualBlock
# from rlcard.games.tractors.models.common_model import PlayerEncoder

# Add Transformer components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output

'''出牌模型'''

class Actor(nn.Module):
    """
    通过状态编码器编码场面信息。
    出牌一般由固定牌+垫牌组成，固定牌为必须出的牌，垫牌为可以随意出的牌。例如在节拖拉机的牌时，若花色中只有一对牌和若干单牌，则一对牌为固定牌，另外还需要从剩余的牌中取出两张单牌，此为垫牌
    动作预测的输入是状态编码+N种固定牌组合+可垫牌，通过分层强化学习预测每张可垫牌的概率+固定牌概率
    """

    def __init__(self, obs_dim, hand_cards_dim, deck_cards_dim=0) -> None:
        super().__init__()
        '''
        @param obs_dim 表示场面的特征向量维度
        @param deck_cards_dim 一副扑克牌的向量维度
        @param hand_cards_dim 玩家手牌的向量维度
        '''

        '''
        状态编码器
        
        历史出牌编码网络，只输入最近15回合的 历史出牌
        使用resnet提取出牌特征，再输入lstm提取出牌历史时序信息。
        这个网络设计有效结合了卷积神经网络的空间特征提取能力和循环神经网络的时序建模能力，
        特别适合处理像牌局序列这样具有时间依赖性的结构化数据。通过ResNet处理每个时间步的2D特征，再通过LSTM整合时序信息，模型能够捕捉牌局间的复杂动态关系。

        
        '''
        #玩家信息编码器
        # self.player_encoder = PlayerEncoder()

        #历史出牌时序特征, 112维手牌特征+4维座位号特征+2维阵营特征
        self.lstm = nn.LSTM(112+4+2, 96, batch_first=True)

        # 手牌特征提取器 (2,4,14) -> 112维
        self.card_encoder = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=3)

        # 当前轮次的出牌信息        
        self.round_play_encoder = nn.Sequential(
            nn.Linear(112+4+2, 112+4+2),
            nn.ReLU(),
            nn.Linear(112+4+2, 256),
            nn.LayerNorm(256)
        )
        
        # 状态投影层（用于点积注意力）        
        self.state_fusion_net = nn.Sequential(
            nn.Linear(108, obs_dim),
            nn.ReLU(),
            nn.Linear(obs_dim, obs_dim),
            nn.LayerNorm(obs_dim)
        )

        # 固定动作编码器
        self.fixed_action_net = nn.Sequential(
            nn.Linear(108, hand_cards_dim),
            nn.ReLU(),
            nn.Linear(hand_cards_dim, hand_cards_dim),
            nn.LayerNorm(hand_cards_dim)
        )

        # 高效的注意力评分网络（双线性形式）
        self.attention = nn.Bilinear(256, 256, 1, bias=False)
        
        # 垫牌动作编码器
        self.action_discard_encoder = nn.Sequential(
            nn.Linear(108, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256)
        )

        # 共享的特征提取模块
        self.shared_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        #动作预测
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征融合模块
        self.fusion_net = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # 注意力机制预测底牌信息，模型可以更好地利用Transformer的注意力机制来预测底牌，考虑到不同牌之间的关系和依赖性，提高底牌预测的准确性。
        self.public_card_transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512),
            num_layers=2
        )
        self.positional_encoding = PositionalEncoding(d_model=256)
        self.public_card_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        self.public_card_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 0: not bottom card, 1: bottom card
        )
        
        # 概率预测头
        self.probability_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Sigmoid()
        )

        # 可学习的标准差 log(σ)，避免数值不稳定
        # self.log_std = nn.Parameter(t.zeros(1, 2*14*4))

    def forward(self, obs_x, actions_fixed, actions_discard) -> Tensor:
        #出牌方位为调整为以自己为0号位
        play_card_history_feat = obs_x['history_play_card']#历史出牌
        play_seat_history_feat = obs_x['history_play_seat']#出牌座位号
        play_team_history_feat = obs_x['history_play_team']#出牌阵营
        round_play_card_mat = obs_x['round_play_card']#当前轮出牌

        '''拼接历史出牌信息进行历史出牌时序预测'''
        play_card_history_enocdefeat = self.card_encoder(play_card_history_feat)
        play_history_feat = t.cat([play_card_history_enocdefeat, play_seat_history_feat, play_team_history_feat], dim=0)
        lstm_out, (h_n, _) = self.lstm(play_history_feat)
        
        '''当前轮次的特征'''
        round_play_cards_encodefeat = self.card_encoder(round_play_card_mat)
        round_play_seat_feat = obs_x['round_play_seat']
        round_play_team_feat = obs_x['round_play_team']
        round_play_feat = self.round_play_encoder(t.cat([round_play_cards_encodefeat, round_play_seat_feat, round_play_team_feat], dim=0))

        '''编码玩家(我的)个人特征'''
        my_seat = obs_x['seat']#我的座位号
        my_team = obs_x['team']#我的阵营
        my_cards = obs_x['hand_cards']#我的手牌
        my_card_enocdefeat = self.card_encoder(my_cards.unsqueeze(0)) +my_seat + my_team
        
        '''上一轮出牌特征信息'''
        last_play_encodefeat = t.cat([self.card_encoder(play_card_history_feat[-1]), play_seat_history_feat[-1], play_team_history_feat[-1]], dim=0)
        last_play_encodefeat = self.round_play_encoder(last_play_encodefeat)
        
        '''场面信息'''
        played_cards = self.card_encoder(obs_x['played_cards'].unsqueeze(0))#已经出过的牌
        level_cards = self.card_encoder(obs_x['level_card'].unsqueeze(0))#当前打第几级，用一副扑克表示[1,4,14]当前的级牌
        score_cards = self.card_encoder(obs_x['score_card'].unsqueeze(0))#当前得分，庄家从80分算起，当闲家得到80分时得分为0，超过80分时得分为负数；闲家从-80分算起，得到80分时得分为0，超过80分时得分为正数，用一副扑克表示（5,10，k）[2,14,4]
        remain_score_cards = self.card_encoder(obs_x['remain_score_card'].unsqueeze(0))#场面剩余分数牌，用两副扑克表示（5,10，k）[2,4,14]
        # combined = t.cat([played_cards, level_cards, score_cards, remain_score_cards], dim=0)
        # combined_feat = self.cards_restnet(combined)#联合计算，提高速度
        
        '''预测八张底牌(n/25 * pre_public_card)'''
        if obs_x['banker'] == my_seat:
            pre_publiccard = obs_x['public_card']
            public_card_feat = obs_x['public_card'].copy().unsqueeze(0)
        else:
            remain_cards = t.zeros(2, 4, 14)
            for _round_play_card_mat in play_card_history_feat:
                for play_card_mat in _round_play_card_mat:
                    remain_cards += play_card_mat.detach()
            for card_mat in round_play_card_mat:
                remain_cards += card_mat.detach()
            remain_cards += my_cards.detach()
            remain_cards = 1-remain_cards
            remain_cards[:, 2:3, 13] = 0
            
            # 原始特征提取
            combined_public_feat = t.cat([self.card_encoder(remain_cards), play_history_feat, round_play_feat], dim=0)
            public_card_encoded = self.card_encoder(combined_public_feat)
            
            # Apply Transformer attention mechanism
            # Reshape to sequence: [batch, channels, height, width] -> [batch, seq_len, channels]
            batch, channels, height, width = public_card_encoded.shape
            public_card_seq = public_card_encoded.view(batch, channels, -1).permute(0, 2, 1)
            
            # Add positional encoding
            public_card_seq = self.positional_encoding(public_card_seq)
            
            # Apply transformer encoder
            transformer_output = self.public_card_transformer(public_card_seq)
            
            # Apply attention mechanism
            attended_output, _ = self.public_card_attention(transformer_output, transformer_output, transformer_output)
            
            # Predict probability for each card position
            card_logits = self.public_card_predictor(attended_output)  # [batch, seq_len, 2]
            card_probabilities = F.softmax(card_logits, dim=-1)[:, :, 1]  # Probability of being a bottom card
            
            # Reshape back to card matrix format
            card_probabilities = card_probabilities.view(batch, height, width)
            
            # Apply mask to only consider remaining cards
            masked_probabilities = card_probabilities * remain_cards  # Apply mask for first card group
            
            # Expand to both card groups
            expanded_probabilities = masked_probabilities.unsqueeze(0).expand(2, -1, -1)
            
            # Sample 8 cards based on probabilities
            flat_probs = expanded_probabilities.reshape(-1)
            sampled_indices = t.multinomial(flat_probs + 1e-8, num_samples=8, replacement=False)
            
            # Create public card feature based on sampled indices
            public_card_feat = t.zeros_like(remain_cards)
            for idx in sampled_indices:
                group_idx = idx // (height * width)
                pos_idx = idx % (height * width)
                row_idx = pos_idx // width
                col_idx = pos_idx % width
                public_card_feat[group_idx, row_idx, col_idx] = 1
                
            public_card_feat *= (1 - my_cards.sum().detach()/25)  # Scale by remaining card ratio
            pre_publiccard = public_card_feat.clone()
            pre_publiccard[pre_publiccard > 0] = 1

        '''融合所有状态特征'''
        fusion_feat = t.cat([lstm_out, last_play_encodefeat, round_play_feat, played_cards, level_cards, score_cards, remain_score_cards, public_card_feat, my_card_enocdefeat], dim=1)
        state_feat = self.state_fusion_net(fusion_feat)

        '''对固定组合牌执行hdmc算法，计算每个action的action value'''
        #状态广播并与固定组合动作牌融合
        state_emb = state_feat.unsqueeze(1).expand(actions_fixed.shape[0], -1)
        #DQN预测固定组合动作牌的自回归值
        fixed_action_prob = self.fixed_action_net(state_emb)
        

        #通过obs_feat场面信息+actions_fixed+actions_discard预测actions_discard中每张牌的出牌概率
        # actions_fixed_feat = self.card_encoder(actions_fixed)
        # actions_discard_feat = self.card_encoder(actions_discard)
        
        

        discard_action_prob = t.empty(actions_discard.shape)
        #以 固定组合动作牌+状态 作为输入，预测剩余待选择牌中每张牌被选择为垫牌的概率，可采用注意力机制
        card_count = obs_x['round_play_card'][-1].sum()
        for i in range(actions_fixed.shape[0]):
            comb_feat = t.cat([state_feat, actions_fixed[i, :, :, :], actions_discard], 1) #[b, n_channels, w, h]
            comb_feat = self.card_encoder(comb_feat)
            # 融合特征处理
            fused = self.fusion_net(comb_feat)  # [batch, 64, 2]
            # 上采样回原始空间尺寸 [batch, 64, 2] -> [batch, 64, 14, 4]
            upsampled = F.interpolate(fused, size=(14, 4), mode='bilinear', align_corners=False)
            # 垫牌的概率预测
            probability = self.probability_head(upsampled)  # [batch, 2, 14, 4]
            # action mask
            probability_each_card = probability*actions_discard
            discard_action_prob[i] = probability_each_card

        return fixed_action_prob, discard_action_prob, pre_publiccard

    #PPO 中使用两个策略分布（一个选固定牌，一个选垫牌）组合成一个完整的动作，并进行训练更新

# Rest of the classes remain unchanged...
class Critic(nn.Module):
    """
    Critic Network, it takes the states as an input,
    and outputs a scalar which indicates the value of the state
    """

    def __init__(self, n_states: int, n_hiddens=256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        forward procedure of the critic
        """
        return self.net.forward(x)

class PPOClip(nn.Module):
    # ... (keep existing implementation)
    pass

class PlayModel(nn.Module):
    def __init__(self):
        super().__init__()

    def load_checkpoint(self):
        pass