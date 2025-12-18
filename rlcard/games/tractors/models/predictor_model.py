import torch
import torch.nn as nn
import math
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
# from rlcard.games.tractors.models.common_model import PlayerEncoder
from rlcard.games.tractors.models.stractor_resnet import ResNet, ResidualBlock
torch.autograd.set_detect_anomaly(True)  # 启用异常检测

# ---------------------------
# Helpers
# ---------------------------
def to_torch(x, dtype=torch.float32, device=torch.device('cpu')):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device).type(dtype)
    if isinstance(x, torch.Tensor):
        return x.to(device).type(dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def entropy_of_probs(p: torch.Tensor, eps=1e-12):
    # p: [B, C], probabilities
    return -torch.sum(p * torch.log(p + eps), dim=-1)  # [B]

# ---------------------------
# Attention Mechanism
# ---------------------------
class AttentionLayer(nn.Module):
    """
    专为博弈历史设计的 Attention：
    - 输入: [B, T, D] (LSTM 输出)
    - 输出: [B, D]   (整段历史的加权表示)
    """
    def __init__(self, input_dim, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or input_dim

        # Key / Value
        self.key = nn.Linear(input_dim, attn_dim, bias=False)
        self.value = nn.Linear(input_dim, input_dim, bias=False)

        # 全局可学习 Query（这是关键）
        self.query = nn.Parameter(torch.randn(1, 1, attn_dim))

        # 缩放
        self.scale = attn_dim ** -0.5

        # 稳定性
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, mask=None):
        """
        x:    [B, T, D]
        mask: [B, T]  (可选，True 表示有效时间步)
        """
        B, T, _ = x.shape

        K = self.key(x)              # [B, T, A]
        V = self.value(x)            # [B, T, D]
        Q = self.query.expand(B, -1, -1)  # [B, 1, A]

        # attention score
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, 1, T]

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        weights = torch.softmax(scores, dim=-1)  # [B, 1, T]

        # 加权求和
        out = torch.bmm(weights, V).squeeze(1)   # [B, D]

        # 残差 + 归一化（非常重要）
        out = self.norm(out + x.mean(dim=1))

        return out

# ---------------------------
# Predictor: outputs marginals + selected important features
# ---------------------------
class Predictor(nn.Module):
    """
    输入: state_feat [B, OBS_DIM]
    输出:
      - opp_marginals: list length NUM_OPPONENTS of [B, 2, 4, 15] (sigmoid)
      - bottom_marginal: [B, 2, 4, 15] (sigmoid)
      - important_feats: [B, K] e.g. probabilities for set of important predicates:
          [opp_has_scorecard, opp_has_bigcard, opp_has_suit_xyz, ...] per opponent aggregated or global
    """
    def __init__(self):
        super().__init__()
        self.num_opps = 4
        hidden_dim = 512
        

        # 手牌特征提取器 (2,4,15) -> (hidden_channels,2,4,5) 
        ##kernelsize=3, padding=1, stride=1以保存卷积后的尺寸不变化
        self.card_encoder = ResNet(ResidualBlock, layers = [2,2,2,2 ], hidden_channels=[14,28,56,112], in_channels=2,out_dim=hidden_dim, kernel_size=3, padding=1, stride=1)
        
        total_predictmodel_params = sum(p.numel() for p in self.card_encoder.parameters())
        print("card_encoder parameters:", total_predictmodel_params)

        #历史出牌时序特征, 接着 card_encoder 输出的out_channels*4*15维出牌特征 + 4维座位号特征
        self.lstm = nn.LSTM(hidden_dim+4, hidden_dim, batch_first=True)
        
        '''在LSTM层后添加三个独立的注意力层：
        self.attention_play：用于处理出牌历史序列
        self.attention_bid：用于处理叫主历史序列
        self.attention_round：用于处理当前回合序列
        
        新增注意力机制模块：

        添加了AttentionLayer类，实现了标准的缩放点积注意力机制
        该模块包含查询（Query）、键（Key）和值（Value）的线性变换
        使用softmax计算注意力权重，并对值进行加权求和

        注意力机制可以帮助模型更好地关注历史序列中更重要的时间步，而不是仅仅依赖最后一个时间步的信息。这应该能提高模型对历史信息的理解能力，特别是对于 tractor（拖拉机）这种需要长期记忆和推理的游戏。

        每个注意力层都会：

        对LSTM输出的序列应用线性变换得到Q、K、V矩阵
        计算注意力分数并应用softmax归一化
        使用注意力权重对值进行加权求和
        返回序列的整体表示，而非单一时间步的表示
        
        这样修改后，模型可以更好地捕捉历史出牌、叫 chủ和回合信息中的重要模式，有助于提高对手牌分布预测的准确性。

        '''
        #self.attention_play = AttentionLayer(hidden_dim*2, hidden_dim)
        self.attention_play = AttentionLayer( hidden_dim)
        self.attention_bid = AttentionLayer( hidden_dim)
        self.attention_round = AttentionLayer( hidden_dim)


        # Assuming card_count = 2*4*15 = 112
        self.card_shape = (2, 4, 15)
        card_count = 2*4*15
        f_dim = 8552# f_dim是所有特征提取之后展开的长度
        opp_head_layers = []
        for _ in range(self.num_opps-1):
            layers = []
            layers.append(nn.Linear(f_dim, f_dim//2))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(f_dim//2, f_dim//4))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(f_dim//4, f_dim//8))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(f_dim//8, 4))
            opp_head_layers.append(nn.Sequential(*layers))

        #预测对手牌中的花色分布（各花色牌的数量），预测其他三家的
        self.opp_heads = nn.ModuleList(opp_head_layers)

        #预测底牌分数
        layers = []
        layers.append(nn.Linear(f_dim, f_dim//2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(f_dim//2, f_dim//4))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(f_dim//4, f_dim//8))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(f_dim//8, 1))
        self.bottom_head = nn.Sequential(*layers)

        # important features head (multi-label), e.g. 16 dims
        # self.important_head = nn.Linear(hidden_dim, k_important)
        # optional small critic for internal eval (not used necessarily)
        
    # def forward1(self, state_feat, isTrain = None):
    #     play_card_history_feat = state_feat['history_play_card']#历史出牌
    #     play_seat_history_feat = state_feat['history_play_seat']#历史出牌座位号
    #     played_card_history_feat=state_feat['history_played_card']#已出过的牌
    #     bid_card_history_feat = state_feat['history_bid_card']#報主記錄
    #     bid_seat_history_feat = state_feat['history_bid_seat']#報主座位号
    #     #当前回合牌
    #     play_card_round_feat = state_feat['round_play_card']
    #     play_seat_round_feat = state_feat['round_play_seat']

    #     score_card_feat = state_feat['score_card']#分數牌
    #     score_remain_card_feat = state_feat['remain_score_card']#分數牌
    #     my_seat_feat = state_feat['my_seat']#我的座位号
    #     banker_seat_feat = state_feat['banker_seat']#庄家号
    #     if isTrain:
    #         player_hand_card_num = state_feat['hand_card'].sum(dim=(2,3,4))#玩家剩余手牌数

    #     #底牌，只有庄家知道
    #     public_card_feat = state_feat['public_card']
    #     seat_equal_mask = (my_seat_feat == banker_seat_feat).sum(1) == 4
    #     expanded_mask = seat_equal_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape: [B, 1, 1, 1]
    #     expanded_mask = expanded_mask.expand_as(public_card_feat)  # Shape: [B, 2, 4, 15]
    #     public_card_feat = public_card_feat * expanded_mask.float()# Zero out public_card_feat where mask is False

    #     #mask可见牌
    #     mask_card = state_feat['history_played_card'].detach() + state_feat['history_bid_card'].sum(dim=1).detach() #[B, seat, 2, 4, 15]
    #     mask_card = mask_card.clamp(min=0.0, max=1.0)
    #     mask_card = 1 - mask_card
    #     mask_card[:,:,1:4,14] = 0.0
        

    #     #多次动作组成的一轮数据，将多次打平成第三维 
    #     b, a, c, d, e, f = play_card_history_feat.shape

    #     # #test code 检查数据合法性
    #     # for i in range(b):
    #     #     #校验出牌历史    
    #     #     played_card = torch.zeros_like(play_card_history_feat[0,0,0]).to(self._device)
    #     #     for j in range(a):
    #     #         #出牌
    #     #         his_play_card_cnt = play_card_history_feat[i,j,0].sum().item()
    #     #         for k in range(4):
    #     #             if his_play_card_cnt != play_card_history_feat[i,j,k].sum().item() or his_play_card_cnt < 1.0:
    #     #                 raise KeyError(play_card_history_feat[i,j])#bug
    #     #             his_play_card_cnt = play_card_history_feat[i,j,k].sum().item()
    #     #             played_card = played_card + play_card_history_feat[i,j,k]
    #     #             #座位号
    #     #             his_play_seat_cnt = play_seat_history_feat[i,j,k].sum().item()
    #     #             if his_play_seat_cnt != 1:
    #     #                 raise KeyError(play_seat_history_feat[i,j])#bug
            
    #     #     #校验叫主
    #     #     for j in range(bid_card_history_feat.shape[1]):
    #     #         if bid_card_history_feat[i][j].sum() == 0 and bid_seat_history_feat[i][j].sum() > 0:
    #     #             raise KeyError(bid_card_history_feat[i,j])#bug
    #     #         if bid_card_history_feat[i][j].sum() > 0 and bid_seat_history_feat[i][j].sum() == 0:
    #     #             raise KeyError(bid_seat_history_feat[i,j])#bug

    #     #     #校验回合牌
    #     #     for j in range(play_card_round_feat.shape[1]):
    #     #         if play_card_round_feat[i][j].sum() == 0 and play_seat_round_feat[i][j].sum() > 0:
    #     #             raise KeyError(play_card_round_feat[i,j])#bug
    #     #         if play_card_round_feat[i][j].sum() > 0 and play_seat_round_feat[i][j].sum() == 0:
    #     #             raise KeyError(play_seat_round_feat[i,j])#bug
    #     #         played_card = played_card + play_card_round_feat[i,j]

    #     #     # 校验已出过的牌
    #     #     if play_card_history_feat[i][14][0].sum() == 0 and (played_card_history_feat[i] != played_card).sum().long() > 0:
    #     #         raise KeyError(played_card_history_feat[i])#bug
            
    #     #     #校验分數牌
    #     #     score_card = score_card_feat[i] + score_remain_card_feat[i]
    #     #     if (score_card.long().sum() != 24 or score_card == 1.0).sum() != 24:
    #     #         raise KeyError(score_card_feat[i][i])#bug
            
    #     #     #校验座位号
    #     #     if my_seat_feat[i].sum().long() != 1 or banker_seat_feat[i].sum().long() != 1:
    #     #         raise KeyError(score_card_feat[i][i])#bug

    #     #     #校验底牌
    #     #     if(state_feat['public_card'][i].sum().long() != 8):
    #     #         raise KeyError(state_feat['public_card'][i])
            
    #     #     #校验所有牌
    #     #     if (played_card_history_feat[i] + mask_card[i] + bid_card_history_feat[i][0] + bid_card_history_feat[i][1]).clamp(min=0.0, max=1.0).sum().long() != 108:
    #     #         raise KeyError(public_card_feat[i])



    #     play_card_history_feat = play_card_history_feat.reshape(b, a*c, d, e, f)
    #     b, a, c, d = play_seat_history_feat.shape
    #     play_seat_history_feat = play_seat_history_feat.reshape(b, a*c, d)
    #     b, a, c, d, e = bid_card_history_feat.shape

    #     bid_card_history_feat = bid_card_history_feat.reshape(b, a, c, d, e)
    #     b, a, c = bid_seat_history_feat.shape
    #     bid_seat_history_feat = bid_seat_history_feat.reshape(b, a, c)
    #     b, a, c, d, e = play_card_round_feat.shape
    #     play_card_round_feat = play_card_round_feat.reshape(b, a, c, d, e)
    #     b, a, c = play_seat_round_feat.shape
    #     play_seat_round_feat = play_seat_round_feat.reshape(b, a, c)
        
        
        
    #     # play_card_history_enocdefeat = self.card_encoder(play_card_history_feat)#一个card_encoder花费40ms

    #     card_feature_dict = {
    #         'play_history': play_card_history_feat.reshape(-1, 2, 4, 15),
    #         'bid_history': bid_card_history_feat.reshape(-1, 2, 4, 15),
    #         'round_history': play_card_round_feat.reshape(-1, 2, 4, 15),
    #         'played_history': played_card_history_feat,
    #         # 'score_card': score_card_feat,
    #         # 'score_remain': score_remain_card_feat,
    #         'public_card': public_card_feat,
    #         'mask_card': mask_card
    #     }

        
    #     # Concatenate all features along batch dimension
    #     batched_features = torch.cat(list(card_feature_dict.values()), dim=0)

    #     # Single encoder call
    #     encoded_features = self.card_encoder(batched_features)

    #     # Split and reshape back
    #     split_sizes = [v.shape[0] for v in card_feature_dict.values()]
    #     split_features = torch.split(encoded_features, split_sizes, dim=0)

    #     # Reconstruct individual features
    #     features_iter = iter(split_features)
    #     play_card_history_enocdefeat = next(features_iter).reshape(
    #         play_card_history_feat.shape[0], play_card_history_feat.shape[1], -1)
    #     bid_card_history_encodefeat = next(features_iter).reshape(
    #         bid_card_history_feat.shape[0], bid_card_history_feat.shape[1], -1)
    #     round_card_history_encodefeat = next(features_iter).reshape(
    #         play_card_round_feat.shape[0], play_card_round_feat.shape[1], -1)
    #     played_card_history_encodefeat = next(features_iter)
    #     # score_card_encodefeat = next(features_iter)
    #     # score_remain_card_encodefeat = next(features_iter)
    #     public_card_encodefeat = next(features_iter)
    #     mask_card_encodefeat = next(features_iter)
    #     '''将原来4个人轮流出的动作展平，4个人各出一个动作为一个回合
    #     再将原来4个人的座位号展平，再将两个特征拼接在一起，
    #     也可以使用交替凭借，
    #     不过，需要注意的是，在实际应用场景中，交替拼接可能不如传统的特征拼接有效，因为：
    #     语义分离：交替拼接可能会破坏特征原有的语义结构
    #     网络学习难度增加：神经网络可能更难从交错特征中学习模式
    #     维度不匹配问题：如果两个张量的最后一维大小不同，交错拼接会更加复杂
    #     这种拼接方式让LSTM能够：
    #     独立学习卡牌出牌和座位位置的时间依赖关系
    #     必要时分别关注卡牌特征和座位特征
    #     清晰区分不同类型的特征'''
    #     # play_card_history_enocdefeat = play_card_history_enocdefeat.reshape(int(play_card_history_enocdefeat.shape[0]/self.num_opps), self.num_opps, -1)
    #     play_history_feat = torch.cat([play_card_history_enocdefeat, play_seat_history_feat], dim=-1)
    #     h_play, (_, _) = self.lstm(play_history_feat)
    #     # 使用注意力机制替代直接取最后一个时间步
    #     h_play = self.attention_play(h_play) # [B, hidden_dim*2] -> [B, hidden_dim*2]

    #     # bid_card_history_encodefeat = self.card_encoder(bid_card_history_feat)
    #     bid_history_feat = torch.cat([bid_card_history_encodefeat, bid_seat_history_feat], dim=-1)
    #     h_bid, (_, _) = self.lstm(bid_history_feat)
    #     # 使用注意力机制
    #     h_bid = self.attention_bid(h_bid) # [B, hidden_dim*2]

    #     # round_card_history_encodefeat = self.card_encoder(play_card_round_feat)
    #     round_history_feat = torch.cat([round_card_history_encodefeat, play_seat_round_feat], dim=-1)
    #     h_round, (_, _) = self.lstm(round_history_feat)
    #     # 使用注意力机制
    #     h_round = self.attention_round(h_round) # [B, hidden_dim*2]


    #     # played_card_history_encodefeat = self.card_encoder(played_card_history_feat)
    #     # score_card_encodefeat = self.card_encoder(score_card_feat)
    #     # score_remain_card_encodefeat = self.card_encoder(score_remain_card_feat)

    #     # public_card_encodefeat = self.card_encoder(public_card_feat)
        

        
    #     # mask_card_encodefeat = self.card_encoder(mask_card)

    #     #特征融合
    #     batch_size = my_seat_feat.shape[0]
    #     # h_play = h_play.reshape(batch_size, -1)
    #     # h_round = h_round.reshape(batch_size, -1)
    #     # h_bid = h_bid.reshape(batch_size, -1)
        
    #     in_feat = torch.cat([h_play, h_round, h_bid, played_card_history_encodefeat, public_card_encodefeat, 
    #                         #  score_card_encodefeat, score_remain_card_encodefeat, 
    #                          mask_card_encodefeat, my_seat_feat, banker_seat_feat], dim=-1)
        
    #     #这两个加起来也就12ms左右
    #     opp_logits = torch.stack([head(in_feat) for head in self.opp_heads], dim=0)  # [num_opps, B, CARD_COUNT]#预测4个玩家
    #     opp_logits.transpose_(1,0)

    #     #对超过手牌数的回归值上线进行截断
    #     # if isTrain:
    #     #     player_hand_card_num = player_hand_card_num.unsqueeze(-1).expand(-1, -1, opp_logits.shape[-1])
    #     #     opp_logits.clamp_(max=player_hand_card_num)
            

    #     bottom_logits = self.bottom_head(in_feat)
    #     bottom_logits.squeeze_(1)
        
    #     return opp_logits, bottom_logits


    #     # 直接预测手牌分布
    #     # in_feat = torch.cat([h_play, h_round, h_bid, played_card_history_encodefeat, public_card_encodefeat, 
    #     #                      score_card_encodefeat, score_remain_card_encodefeat, my_seat_feat, banker_seat_feat], dim=-1)
    #     # opp_logits = torch.stack([head(in_feat) for head in self.opp_heads], dim=0)  # [num_opps, B, CARD_COUNT]#预测4个玩家
    #     # bottom_logits = self.bottom_head(in_feat)  # [B, output_dim]
    #     # # important_logits = self.important_head(h)           # [B, K]
    #     # # Update the probability calculation accordingly:
    #     # opp_probs = torch.sigmoid(opp_logits).view(self.num_opps, -1, *self.card_shape)  # [num_opps, B, 2, 4, 15]#智能体预测4家手牌的概率
    #     # bottom_prob = torch.sigmoid(bottom_logits).view(-1, *self.card_shape)             # [B, 2, 4, 15]
    #     # # important_prob = torch.sigmoid(important_logits)       # multi-label probabilities

        
        
    #     # # enumerate(state_feat['history_play_card'])
    #     # # for bat, player_play_cards in enumerate(state_feat['history_play_card']):
    #     # #     for seat_idx, play_card in enumerate(player_play_cards):
    #     # #         mask_card[play_seat_history_feat[round_idx][seat_idx]] += play_card
        
        
    #     # # opp_probs = torch.from_numpy(np.array([opp_prob*mask_card for opp_prob in opp_probs])).to(self._device)
    #     # # Transpose to get the desired shape [B, num_opps, 2, 4, 15]:
    #     # opp_probs = opp_probs*mask_card#mask_card自动广播到opp_probs相同的形状
    #     # opp_probs = torch.transpose(opp_probs, 0, 1)
    #     # bottom_prob = bottom_prob*mask_card
        
        
    #     # return opp_probs, bottom_prob

    def forward(self, state_feat, isTrain = None):
        B = state_feat['history_play_seat'].shape[0]#批次
        play_card_history_feat = state_feat['history_play_card']#历史出牌
        play_seat_history_feat = state_feat['history_play_seat']#历史出牌座位号
        played_card_history_feat=state_feat['history_played_card']#已出过的牌
        bid_card_history_feat = state_feat['history_bid_card']#報主記錄
        bid_seat_history_feat = state_feat['history_bid_seat']#報主座位号
        #当前回合牌
        play_card_round_feat = state_feat['round_play_card']
        play_seat_round_feat = state_feat['round_play_seat']

        score_card_feat = state_feat['score_card']#分數牌
        score_remain_card_feat = state_feat['remain_score_card']#分數牌
        my_seat_feat = state_feat['my_seat']#我的座位号
        banker_seat_feat = state_feat['banker_seat']#我的座位号
        mask_cards = state_feat['mask_card']#.copy()
        hand_card_feat = state_feat['hand_card']#.copy()
        public_card_feat = state_feat['public_card']
        
        # #底牌，只有庄家知道
        # 
        # seat_equal_mask = (my_seat_feat == banker_seat_feat).sum(1) == 4
        # expanded_mask = seat_equal_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape: [B, 1, 1, 1]
        # expanded_mask = expanded_mask.expand_as(public_card_feat)  # Shape: [B, 2, 4, 15]
        # public_card_feat = public_card_feat * expanded_mask.float()# Zero out public_card_feat where mask is False
        # if isTrain:
        #     player_hand_card_num = state_feat['hand_card'].sum(dim=(2,3,4))#玩家剩余手牌数
            # player_hand_card_num = player_hand_card_num.unsqueeze(-1).expand(-1, -1, opp_logits.shape[-1])
            # opp_logits.clamp_(max=player_hand_card_num)
        
        
        # #test code 检查数据合法性
        # for i in range(play_card_history_feat.shape[0]):
        #     #校验出牌历史    
        #     played_card = torch.zeros_like(play_card_history_feat[0,0,0]).to(self._device)
        #     for j in range(play_card_history_feat.shape[1]):
        #         #出牌
        #         his_play_card_cnt = play_card_history_feat[i,j,0].sum().item()
        #         for k in range(4):
        #             if his_play_card_cnt != play_card_history_feat[i,j,k].sum().item():
        #                 raise KeyError(play_card_history_feat[i,j])#bug
        #             his_play_card_cnt = play_card_history_feat[i,j,k].sum().item()
        #             played_card = played_card + play_card_history_feat[i,j,k]
        #             #座位号
        #             his_play_seat_cnt = play_seat_history_feat[i,j,k].sum().item()
        #             if his_play_seat_cnt != 1 and his_play_card_cnt != 0:
        #                 raise KeyError(play_seat_history_feat[i,j])#bug
            
        #     #校验叫主
        #     for j in range(bid_card_history_feat.shape[1]):
        #         if bid_card_history_feat[i][j].sum() == 0 and bid_seat_history_feat[i][j].sum() > 0:
        #             raise KeyError(bid_card_history_feat[i,j])#bug
        #         if bid_card_history_feat[i][j].sum() > 0 and bid_seat_history_feat[i][j].sum() == 0:
        #             raise KeyError(bid_seat_history_feat[i,j])#bug

        #     #校验回合牌
        #     for j in range(play_card_round_feat.shape[1]):
        #         if play_card_round_feat[i][j].sum() == 0 and play_seat_round_feat[i][j].sum() > 0:
        #             raise KeyError(play_card_round_feat[i,j])#bug
        #         if play_card_round_feat[i][j].sum() > 0 and play_seat_round_feat[i][j].sum() == 0:
        #             raise KeyError(play_seat_round_feat[i,j])#bug
        #         played_card = played_card + play_card_round_feat[i,j]

        #     # 校验已出过的牌
        #     if play_card_history_feat[i][14][0].sum() == 0 and (played_card_history_feat[i] != played_card).sum().long() > 0:
        #         raise KeyError(played_card_history_feat[i])#bug
            
        #     # 校验牌总数
        #     if (played_card_history_feat[i].sum().item() + state_feat['hand_card'][i].sum().item() + 8) != 108:
        #         raise KeyError(played_card_history_feat[i])#bug
            
        #     #校验分數牌
        #     score_card = score_card_feat[i] + score_remain_card_feat[i]
        #     if (score_card.long().sum() != 24 or score_card == 1.0).sum() != 24:
        #         raise KeyError(score_card_feat[i][i])#bug
            
        #     #校验座位号
        #     if my_seat_feat[i].sum().long() != 1 or banker_seat_feat[i].sum().long() != 1:
        #         raise KeyError(score_card_feat[i][i])#bug

        #     #校验底牌
        #     if(state_feat['public_card'][i].sum().long() != 8):
        #         raise KeyError(state_feat['public_card'][i])
            
            
            


        #多次动作组成的一轮数据，将多次打平成第三维 
        b, a, c, d, e, f = play_card_history_feat.shape
        play_card_history_feat = play_card_history_feat.reshape(b, -1)

        b, a, c, d = play_seat_history_feat.shape
        play_seat_history_feat = play_seat_history_feat.reshape(b, -1)

        b, a, c, d, e = bid_card_history_feat.shape
        bid_card_history_feat = bid_card_history_feat.reshape(b, -1)

        b, a, c = bid_seat_history_feat.shape
        bid_seat_history_feat = bid_seat_history_feat.reshape(b, -1)

        b, a, c, d, e = play_card_round_feat.shape
        play_card_round_feat = play_card_round_feat.reshape(b,  -1)

        b, a, c = play_seat_round_feat.shape
        play_seat_round_feat = play_seat_round_feat.reshape(b, -1)
        
        score_card_feat = score_card_feat.reshape(b, -1)
        
        
        
        # play_card_history_enocdefeat = self.card_encoder(play_card_history_feat)#一个card_encoder花费40ms

        #其他玩家的维度
        my_seat_idx = my_seat_feat.argmax(dim=1)
        other_seat_feats = torch.stack(
            [F.one_hot((my_seat_idx + k) % 4, num_classes=4) for k in (1, 2, 3)],
            dim=0
        )

        #自身手牌批次
        my_hand_cards = []
        for b in range(B):
            mycard = hand_card_feat[b][my_seat_feat[b].argmax()]
            if banker_seat_feat[b].argmax() == my_seat_idx[b]:#如果自己是庄家，再mask掉自身底牌
                mycard = mycard +  public_card_feat[b]
                mycard[mycard>1] = 1
            my_hand_cards.append(mycard)
        my_hand_cards = torch.stack(my_hand_cards, dim=0).to(device=self._device)
        

        opp_logits = []
        for i in range(len(other_seat_feats)):
            other_seat = other_seat_feats[i]
            cur_mask = []#第i个玩家，批次中的mask，
            
            for b in range(other_seat.shape[0]):#other_seat.shape[0]就是批次
                s = other_seat[b].argmax()
                m = mask_cards[b][s] - my_hand_cards[b]#并且去掉自身的手牌
                m[m<0] = 0
                cur_mask.append(m)
            cur_mask = torch.stack(cur_mask).to(device=self._device)

                   
            in_feat = torch.cat([bid_card_history_feat, bid_seat_history_feat, 
                play_card_history_feat, play_seat_history_feat,
                play_card_round_feat, play_seat_round_feat, 
                my_hand_cards.reshape(B, -1), my_seat_feat,
                score_card_feat, 
                cur_mask.reshape(B, -1), #这个特征影响大
                other_seat_feats[i]], dim=-1)
            opp_logit = self.opp_heads[i](in_feat)
            cur_mask = cur_mask.any(dim=-1).sum(dim=-2).int()
            cur_mask[cur_mask>1] = 1
            opp_logit = opp_logit * cur_mask
            opp_logits.append(opp_logit)

        opp_logits = torch.stack(opp_logits, dim=0)  # [num_opps, B, CARD_COUNT]#预测4个玩家
        opp_logits.transpose_(1,0)

        bottom_logits = self.bottom_head(in_feat)
        bottom_logits.squeeze_(1)
        
        return opp_logits, bottom_logits


        # 直接预测手牌分布
        # in_feat = torch.cat([h_play, h_round, h_bid, played_card_history_encodefeat, public_card_encodefeat, 
        #                      score_card_encodefeat, score_remain_card_encodefeat, my_seat_feat, banker_seat_feat], dim=-1)
        # opp_logits = torch.stack([head(in_feat) for head in self.opp_heads], dim=0)  # [num_opps, B, CARD_COUNT]#预测4个玩家
        # bottom_logits = self.bottom_head(in_feat)  # [B, output_dim]
        # # important_logits = self.important_head(h)           # [B, K]
        # # Update the probability calculation accordingly:
        # opp_probs = torch.sigmoid(opp_logits).view(self.num_opps, -1, *self.card_shape)  # [num_opps, B, 2, 4, 15]#智能体预测4家手牌的概率
        # bottom_prob = torch.sigmoid(bottom_logits).view(-1, *self.card_shape)             # [B, 2, 4, 15]
        # # important_prob = torch.sigmoid(important_logits)       # multi-label probabilities

        
        
        # # enumerate(state_feat['history_play_card'])
        # # for bat, player_play_cards in enumerate(state_feat['history_play_card']):
        # #     for seat_idx, play_card in enumerate(player_play_cards):
        # #         mask_card[play_seat_history_feat[round_idx][seat_idx]] += play_card
        
        
        # # opp_probs = torch.from_numpy(np.array([opp_prob*mask_card for opp_prob in opp_probs])).to(self._device)
        # # Transpose to get the desired shape [B, num_opps, 2, 4, 15]:
        # opp_probs = opp_probs*mask_card#mask_card自动广播到opp_probs相同的形状
        # opp_probs = torch.transpose(opp_probs, 0, 1)
        # bottom_prob = bottom_prob*mask_card
        
        
        # return opp_probs, bottom_prob
    

    def toDevice(self, device):
        self._device = device
        self.to(device)
        self.card_encoder.toDevice(device)

