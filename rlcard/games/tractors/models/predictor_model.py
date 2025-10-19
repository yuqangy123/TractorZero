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
    def __init__(self, hidden_dim, num_opps):
        super().__init__()
        self.num_opps = num_opps
        #历史出牌时序特征, 2*4*15维出牌特征+4维座位号特征
        self.lstm = nn.LSTM(112+4, hidden_dim, batch_first=True)

        # 手牌特征提取器 (2,4,15) -> 112维
        self.card_encoder = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=3)

        # Assuming card_count = 2*4*15 = 112
        self.card_shape = (2, 4, 15)
        card_count = 2*4*15
        self.opp_heads = nn.ModuleList([nn.Linear(hidden_dim, card_count) for _ in range(num_opps)])
        self.bottom_head = nn.Linear(hidden_dim, card_count)

        # important features head (multi-label), e.g. 16 dims
        # self.important_head = nn.Linear(hidden_dim, k_important)
        # optional small critic for internal eval (not used necessarily)
        
    def forward(self, state_feat):
        # h = self.obs_proj(state_feat)         # [B, hidden_dim]
        play_card_history_feat = state_feat['history_play_card']#历史出牌
        play_seat_history_feat = state_feat['history_play_seat']#历史出牌座位号
        bid_card_history_feat = state_feat['history_bid_card']#報主記錄
        bid_seat_history_feat = state_feat['history_bid_seat']#報主座位号
        score_card_feat = state_feat['score_card']#分數牌

        
        play_card_history_enocdefeat = self.card_encoder(play_card_history_feat)
        play_history_feat = torch.cat([play_card_history_enocdefeat, play_seat_history_feat], dim=0)
        h_play, (h_n, _) = self.lstm(play_history_feat)

        bid_card_history_enocdefeat = self.card_encoder(bid_card_history_feat)
        bid_history_feat = torch.cat([bid_card_history_enocdefeat, bid_seat_history_feat], dim=0)
        h_bid, (h_n, _) = self.lstm(bid_history_feat)

        score_card_encodefeat = self.card_encoder(score_card_feat)

        in_feat = torch.cat([h_play, h_bid, score_card_encodefeat])

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