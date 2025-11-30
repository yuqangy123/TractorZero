import torch
import torch.nn as nn
import math
import numpy as np
import torch.optim as optim
# from rlcard.games.tractors.models.common_model import PlayerEncoder
from rlcard.games.tractors.models.stractor_resnet import ResNet, ResidualBlock


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
        hidden_dim = 256
        

        # 手牌特征提取器 (2,4,15) -> (hidden_channels,2,4,5) 
        ##kernelsize=3, padding=1, stride=1以保存卷积后的尺寸不变化
        self.card_encoder = ResNet(ResidualBlock, [2, 2, 2, 2], hidden_channels=[14,28,56,112], in_channels=2,out_channels=2, kernel_size=3, padding=1, stride=1)


        #历史出牌时序特征, 接着 card_encoder 输出的out_channels*4*15维出牌特征 + 4维座位号特征
        self.lstm = nn.LSTM(2*4*15+4, hidden_dim, batch_first=True)


        # Assuming card_count = 2*4*15 = 112
        self.card_shape = (2, 4, 15)
        card_count = 2*4*15
        # 2924是所有特征提取之后展开的长度
        self.opp_heads = nn.ModuleList([nn.Linear(hidden_dim, card_count) for _ in range(self.num_opps)])
        self.bottom_head = nn.Linear(hidden_dim, card_count)

        # important features head (multi-label), e.g. 16 dims
        # self.important_head = nn.Linear(hidden_dim, k_important)
        # optional small critic for internal eval (not used necessarily)
        
    def forward(self, state_feat):
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

        #多次动作组成的一轮数据，将多次打平成第三维
        b, a, c, d, e = play_card_history_feat.shape
        play_card_history_feat = play_card_history_feat.reshape(b*a, c, d, e)
        b, a, c = play_seat_history_feat.shape
        play_seat_history_feat = play_seat_history_feat.reshape(b*a, c)
        b, a, c, d, e = bid_card_history_feat.shape
        bid_card_history_feat = bid_card_history_feat.reshape(b*a, c, d, e)
        b, a, c = bid_seat_history_feat.shape
        bid_seat_history_feat = bid_seat_history_feat.reshape(b*a, c)
        b, a, c, d, e = play_card_round_feat.shape
        play_card_round_feat = play_card_round_feat.reshape(b*a, c, d, e)
        b, a, c = play_seat_round_feat.shape
        play_seat_round_feat = play_seat_round_feat.reshape(b*a, c)


        play_card_history_enocdefeat = self.card_encoder(play_card_history_feat)
        '''将原来4个人轮流出的动作展平，4个人各出一个动作为一个回合
        再将原来4个人的座位号展平，再将两个特征拼接在一起，
        也可以使用交替凭借，
        不过，需要注意的是，在实际应用场景中，交替拼接可能不如传统的特征拼接有效，因为：
        语义分离：交替拼接可能会破坏特征原有的语义结构
        网络学习难度增加：神经网络可能更难从交错特征中学习模式
        维度不匹配问题：如果两个张量的最后一维大小不同，交错拼接会更加复杂
        这种拼接方式让LSTM能够：
        独立学习卡牌出牌和座位位置的时间依赖关系
        必要时分别关注卡牌特征和座位特征
        清晰区分不同类型的特征'''
        # play_card_history_enocdefeat = play_card_history_enocdefeat.reshape(int(play_card_history_enocdefeat.shape[0]/self.num_opps), self.num_opps, -1)
        play_history_feat = torch.cat([play_card_history_enocdefeat, play_seat_history_feat], dim=-1)
        h_play, (_, _) = self.lstm(play_history_feat)

        bid_card_history_encodefeat = self.card_encoder(bid_card_history_feat)
        bid_history_feat = torch.cat([bid_card_history_encodefeat, bid_seat_history_feat], dim=-1)
        h_bid, (_, _) = self.lstm(bid_history_feat)

        round_card_history_encodefeat = self.card_encoder(play_card_round_feat)
        round_history_feat = torch.cat([round_card_history_encodefeat, play_seat_round_feat], dim=-1)
        h_round, (_, _) = self.lstm(round_history_feat)

        played_card_history_encodefeat = self.card_encoder(played_card_history_feat)
        score_card_encodefeat = self.card_encoder(score_card_feat)
        score_remain_card_encodefeat = self.card_encoder(score_remain_card_feat)

        #特征融合
        h_play = h_play.reshape(my_seat_feat.shape[0], -1)
        h_round = h_round.reshape(my_seat_feat.shape[0], -1)
        h_bid = h_bid.reshape(my_seat_feat.shape[0], -1)
        in_feat = torch.cat([h_play, h_round, h_bid, played_card_history_encodefeat, score_card_encodefeat, score_remain_card_encodefeat, my_seat_feat], dim=-1)

        opp_logits = [head(in_feat) for head in self.opp_heads]   # each [B, seat, CARD_COUNT]
        bottom_logits = self.bottom_head(in_feat)                 # [B, CARD_COUNT]
        # important_logits = self.important_head(h)           # [B, K]
        # convert to probabilities and reshape
        opp_probs = [torch.sigmoid(lg).view(-1, *self.card_shape) for lg in opp_logits]   # each [B, seat, 2, 4, 15]
        bottom_prob = torch.sigmoid(bottom_logits).view(-1, *self.card_shape)             # [B, seat, 2, 4, 15]
        # important_prob = torch.sigmoid(important_logits)       # multi-label probabilities

        #mask可见牌
        mask_card = [torch.zeros(*self.card_shape)]*self.num_opps #[seat, 2, 4, 15]
        for round_idx, player_play_cards in enumerate(play_card_history_feat):
            for seat_idx, play_card in enumerate(player_play_cards):
                mask_card[play_seat_history_feat[round_idx][seat_idx]] += play_card
        
        for round_idx,bid_cards in enumerate(bid_card_history_feat):
            mask_card[bid_seat_history_feat[round_idx]] += bid_cards

        for i in range(len(opp_probs)):
            mask = mask_card[i] > 0
            mask = mask.unsqueeze(0).expand_as(opp_probs[i])  # shape: [B, seat, 2, 4, 15]
            opp_probs[i][mask] = 0
            bottom_prob[i][mask] = 0

        return opp_probs, bottom_prob
    
    def toDevice(self, device):
        self.to(device)
        self.card_encoder.toDevice(device)


# ---------------------------
# Belief feature projector & confidence calculator
# ---------------------------
class BeliefProcessor(nn.Module):
    """
    将高维的 (opp_marginals, bottom_marginal, important_feats) 投影为紧凑向量 belief_repr
    并计算置信度 c in [0,1]
    """
    def __init__(self, card_count, num_opps, k_important, repr_dim):
        super().__init__()
        self.card_count = card_count
        self.num_opps = num_opps
        self.k_important = k_important
        self.repr_dim = repr_dim
        # project concatenated marginals to compact vector
        inp_dim = card_count * (num_opps + 1) + k_important
        self.proj = nn.Sequential(
            nn.Linear(inp_dim, 256),
            nn.ReLU(),
            nn.Linear(256, repr_dim)
        )
        # optional small network to predict a confidence scalar from features
        self.conf_net = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, opp_probs_list, bottom_prob, important_prob, time_step=None, max_steps=None):
        """
        opp_probs_list: list of [B, CARD_COUNT]
        bottom_prob: [B, CARD_COUNT]
        important_prob: [B, K]
        time_step, max_steps: used for time-based confidence scaling (optional)
        returns:
           belief_repr: [B, repr_dim]
           conf: [B] in [0,1]
           entropy_vals: [B] raw entropy measure (for debug)
        """
        B = bottom_prob.shape[0]
        # concat marginals: [B, CARD_COUNT*(num_opps+1)]
        concat = torch.cat([p for p in opp_probs_list] + [bottom_prob], dim=-1)  # [B, C*(num_opps+1)]
        inp = torch.cat([concat, important_prob], dim=-1)   # [B, inp_dim]
        belief_repr = self.proj(inp)

        # entropy-based confidence: average normalized entropy over opps+bottom
        # compute entropy per distribution per batch
        ent_list = []
        for p in opp_probs_list:
            ent = entropy_of_probs(p)    # [B]
            ent_list.append(ent)
        ent_b = entropy_of_probs(bottom_prob)  # [B]
        ent_all = torch.stack(ent_list + [ent_b], dim=0)  # [num_opps+1, B]
        ent_mean = torch.mean(ent_all, dim=0)  # [B]
        # normalize entropy to [0,1] by dividing by max entropy log(C)
        H_max = math.log(self.card_count + 1e-12)
        entropy_norm = ent_mean / (H_max + 1e-12)   # lower is better (less uncertainty)
        entropy_conf = 1.0 - entropy_norm           # map low ent -> high conf

        # also allow learned conf_net (combines features) and blend learned + entropy
        learned_conf = self.conf_net(inp).squeeze(-1)  # [B]
        # combine: weighted average (could be hyperparam)
        conf = 0.6 * entropy_conf + 0.4 * learned_conf

        # time-based scaling: early steps -> downscale confidence
        TIME_CONF_SCALE = 1.0
        if time_step is not None and max_steps is not None:
            # simple schedule: sigmoid centered at mid-game
            t = float(time_step) / float(max_steps)
            time_factor = float(1.0 / (1.0 + math.exp(-12*(t-0.5))))  # steep sigmoid from 0 ->1
            conf = conf * (0.5 + TIME_CONF_SCALE * time_factor * 0.5)  # combine
            # alternative: conf = conf * (t^alpha) etc.

        # clamp
        conf = torch.clamp(conf, 0.0, 1.0)
        return belief_repr, conf, ent_mean