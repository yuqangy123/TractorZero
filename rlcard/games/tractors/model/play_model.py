import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
import numpy as np
import scipy.signal
import gym
import os
import datetime
from .BasicBlockM import ResNet, ResidualBlock
# from .common_model import PlayerEncoder

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

    def __init__(self, obs_dim, goal_dim, out_dim, hidden_dim=256) -> None:
        super().__init__()
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(obs_dim + goal_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, out_dim),
                        nn.Tanh()
                        )

    def forward(self, obs_x, goal) -> Tensor:
        if goal is None:
            return self.actor(obs_x)
        return self.actor(t.cat([obs_x, goal], 1))
    

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

class PPOClip():
    def __init__(self, state_dim, goal_dim, action_dim, device):
        self._device = device
        lr = 0.001
        #obs_dim, goal_dim, out_dim, hidden_dim=256
        self.actor = Actor(state_dim, goal_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)#放在 learner 代码里
        
        self.critic = Critic(state_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)#放在 learner 代码里
        
        self.mseLoss = t.nn.MSELoss()#放在 learner 代码里
    
    def select_action(self, state, goal=None):
        # state = t.FloatTensor(state.reshape(1, -1)).to(self._device)
        # if goal is None:
        #     goal = t.FloatTensor(goal.reshape(1, -1)).to(self._device)
        
        # return self.actor(state, goal).detach().cpu().data.numpy().flatten()
        return self.actor(state, goal)
    
    def update(self, buffer, n_iter, batch_size):
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)
            
            # convert np arrays into tensors
            state = t.FloatTensor(state).to(self._device)
            action = t.FloatTensor(action).to(self._device)
            reward = t.FloatTensor(reward).reshape((batch_size,1)).to(self._device)
            next_state = t.FloatTensor(next_state).to(self._device)
            goal = t.FloatTensor(goal).to(self._device)
            gamma = t.FloatTensor(gamma).reshape((batch_size,1)).to(self._device)
            done = t.FloatTensor(done).reshape((batch_size,1)).to(self._device)
            
            # select next action
            next_action = self.actor(next_state, goal).detach()
            
            # Compute target Q-value:
            target_Q = self.critic(next_state, next_action, goal).detach()
            target_Q = reward + ((1-done) * gamma * target_Q)
            
            # Optimize Critic:
            critic_loss = self.mseLoss(self.critic(state, action, goal), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Compute actor loss:
            actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
                
                
    def save(self, directory, name):
        t.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        t.save(self.critic.state_dict(), '%s/%s_crtic.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(t.load('%s/%s_actor.pth' % (directory, name), map_location='cpu'))
        self.critic.load_state_dict(t.load('%s/%s_crtic.pth' % (directory, name), map_location='cpu'))  

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
        self.query = nn.Parameter(t.randn(1, 1, attn_dim))

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
        scores = t.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, 1, T]

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        weights = t.softmax(scores, dim=-1)  # [B, 1, T]

        # 加权求和
        out = t.bmm(weights, V).squeeze(1)   # [B, D]

        # 残差 + 归一化（非常重要）
        out = self.norm(out + x.mean(dim=1))

        return out


class BankerModel(nn.Module):
    def __init__(self, num_action_type, device):
        super().__init__()
        self._device = device
        self.num_action_type = num_action_type
        self.cards_shape = (2, 4, 15)
        hidden_dim = 512

        # 手牌特征提取器 (2,4,15) -> (hidden_channels,2,4,5) 
        ##kernelsize=3, padding=1, stride=1以保存卷积后的尺寸不变化
        self.cards_encoder = ResNet(ResidualBlock, layers = [2,2,2,2 ], hidden_channels=[14,28,56,112], \
                                        in_channels=2,out_dim=hidden_dim, kernel_size=3, padding=1, stride=1)

        
        #历史出牌时序特征, 接着 cards_encoder 输出的out_channels*4*15维出牌特征 + 4维座位号特征
        # self.lstm = nn.LSTM(hidden_dim+4, hidden_dim, batch_first=True)
        # self.attention_play = AttentionLayer( hidden_dim+4)
        # self.attention_bid = AttentionLayer( hidden_dim+4)
        # self.attention_round = AttentionLayer( hidden_dim+4)

        #上层策略网络
        self.strategy_net = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 7, bias=False),
            nn.Sigmoid()
        )
        #下层出牌牌型网络
        self.action_type_net = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, num_action_type, bias=False),
        )
        #下层出牌Q值网络
        self.action_q_net = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 3, bias=False),
            nn.Sigmoid()
        )
        
        
        # #牌型决策层模型， 输出4种牌型概率
        # self.actionTypeModel = PPOClip(3620, 0, self.action_type_num, device=device)
        # #具体出牌层模型，输出牌型中每个具体牌的回归值
        # self.actionCardModel = PPOClip(3620, self.action_type_num, 1, device=device)
        # #垫牌模层模型，输出需要垫的牌矩阵
        # self.actionDiscardModel = PPOClip(3620, self.action_type_num+2*4*15, 2*4*15, device=device)

    def __is_tractor_batch(self, cnt):
        # cnt: [B,4,15]
        B = cnt.size(0)

        # 1. 用到的花色数量
        suit_used = (cnt.sum(dim=2) > 0).int()     # [B,4]
        suit_count = suit_used.sum(dim=1)          # [B]

        cond_one_suit = (suit_count == 1)

        # 2. 找出那个花色
        suit_idx = suit_used.argmax(dim=1)         # [B]

        # 3. 取该花色的 rank 计数
        batch_idx = t.arange(B, device=cnt.device)
        rank_cnt = cnt[batch_idx, suit_idx]        # [B,15]

        # 4. 必须全是 2（不能有 1）
        cond_all_pairs = ((rank_cnt == 2) | (rank_cnt == 0)).all(dim=1)

        # 5. 至少两个对子
        pair_mask = (rank_cnt == 2)
        pair_num = pair_mask.sum(dim=1)
        cond_min_len = (pair_num >= 2)

        # 6. 连续性判断
        # 取出 rank 下标
        is_continuous = []
        for b in range(B):
            ranks = t.where(pair_mask[b])[0]
            if len(ranks) >= 2 and t.all(ranks[1:] - ranks[:-1] == 1):
                is_continuous.append(True)
            else:
                is_continuous.append(False)

        cond_continuous = t.tensor(is_continuous, device=cnt.device)

        return cond_one_suit & cond_all_pairs & cond_min_len & cond_continuous

    def forward_tp(self, z, x, mask, return_value=False, flags=None):
        x_feat = self.cards_encoder(x)
        #连接x与z，送入action_type_net
        x_feat = x_feat.flatten(1, 2)
        z_expanded = z.unsqueeze(0).expand(self.num_action_type, -1, -1)
        t_feat = [z_expanded[i] for i in range(self.num_action_type)] + [x_feat]
        output = t.cat(t_feat, dim=-1)        
        logits = self.action_type_net(output)
    
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = t.multinomial(mask, num_samples=1).squeeze(1)
        else:
            out = logits * mask
            action = t.argmax(out, dim=0)[0]
            
        if return_value:
            return dict(action=action, values=out)
        else:
            return dict(action=action)

    def forward_act(self, z, x, return_value=False, flags=None):
        x_feat = self.cards_encoder(x)
        output = t.cat([z, z, z, z, x_feat], dim=-1)
        output = t.cat([z, z, z, z, x_feat], dim=-1)
        
        logits = self.action_q_net(output)
        
        win_rate, win, lose = t.split(logits, (1, 1, 1), dim=-1)
        win_rate = t.tanh(win_rate)
        _win_rate = (win_rate + 1) / 2
        out = _win_rate * win + (1. - _win_rate) * lose

        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = t.randint(out.shape[0], (1,))[0]
        else:
            action = t.argmax(out, dim=0)[0]
                
        if return_value:
            return dict(action=action, max_value=t.max(out), values=(win_rate, win, lose))
        else:
            return dict(action=action, max_value=t.max(out))
        
    
    def forward(self, obs, play_action_seq, legal_actions):
        cards_feat = self.cards_encoder(obs)
        
        play_seq_feat = self.cards_encoder(play_action_seq)
        h_play, (_, _) = self.lstm(play_seq_feat)
        # 使用注意力机制替代直接取最后一个时间步
        h_play_att = self.attention_play(play_seq_feat) # [B, hidden_dim*2] -> [B, hidden_dim*2]
        h_play += h_play_att
        
        pub_x = t.cat([cards_feat, h_play], dim=0)
        
        strategy_logits = self.strategy_net(pub_x)
        best_s = t.argmax(strategy_logits, dim=1)
        strategy_feat = t.zeros((7,), device=self._device, dtype=t.float32)
        strategy_feat[best_s] = 1.
        
        input_ty = t.cat([pub_x, strategy_feat], dim=0)
        action_type_logits = self.action_type_net(input_ty)
        best_t = t.argmax(action_type_logits, dim=1)
        
        actions_batch = t.tensor(legal_actions[best_t], dtype=t.float32, device=self._device)
        pub_x_batch = np.repeat( pub_x[np.newaxis, :, :], actions_batch.shape[0], axis=0 )
        strategy_feat_batch = np.repeat( strategy_feat[np.newaxis, :, :], actions_batch.shape[0], axis=0 )
        z_batch = np.concatenate((actions_batch, pub_x_batch, strategy_feat_batch), axis=1)
        act_logits = self.action_q_net(z_batch)
        
        return dict(value=act_logits, strategy=best_s)
    def forward1(self, state_feat, isTrain = None):
        B = state_feat['history_play_seat'].shape[0]#批次
        play_card_history_feat = state_feat['history_play_card']#历史出牌
        play_seat_history_feat = state_feat['history_play_seat']#历史出牌座位号
        played_card_history_feat=state_feat['history_played_card']#已出过的牌
        bid_card_history_feat = state_feat['history_bid_card']#報主記錄
        bid_seat_history_feat = state_feat['history_bid_seat']#報主座位号
        
        play_card_round_feat = state_feat['round_play_card']#当前回合牌
        play_seat_round_feat = state_feat['round_play_seat']#当前回合牌

        score_card_feat = state_feat['score_card']#分數牌
        score_remain_card_feat = state_feat['remain_score_card']#分數牌
        my_seat_feat = state_feat['my_seat']#我的座位号
        banker_seat_feat = state_feat['banker_seat']#我的座位号
        mask_cards = state_feat['mask_card']#.copy()#mask牌
        hand_card_feat = state_feat['hand_card']#.copy()
        public_card_feat = state_feat['public_card']
        legal_actions = state_feat['legal_actions']#合法动作，【单牌，对子，甩牌，拖拉机】4维特征，对应
        action_types = np.array(list(legal_actions.keys()))
        action_cards = list(legal_actions.values())#合法
        
        # #是否首出
        # player0_cards = play_card_round_feat[:,0]#[B,4,2,4,15] -> [B,2,4,15]
        # cnt = player0_cards.sum(dim=1)   # [B,4,15] cnt[b, s, r] ∈ {0,1,2} 第 b 个 batch 中，玩家 0 在 (suit=s, rank=r) 上有几张
        # total_cards = cnt.sum(dim=(1,2))   # [B] 总牌数
        # is_first_play = (total_cards == 0)#首出
        # is_single = (total_cards == 1)
        # is_pair = (total_cards == 2) & (cnt.max(dim=2).values.max(dim=1).values == 2)
        # tractor_mask = self.__is_tractor_batch(cnt)
        # round_play_card_type = t.full((B,), fill_value=-1, device=cnt.device)
        # round_play_card_type[is_single] = 0#单张
        # round_play_card_type[is_pair] = 1#对子
        # round_play_card_type[tractor_mask] = 2#拖拉机
        # round_play_card_type[round_play_card_type == -1] = 3# 剩下的全部是甩牌



        
        ################################################################################

        # #底牌，只有庄家知道
        # public_card_feat = state_feat['public_card']
        # seat_equal_mask = (my_seat_feat == banker_seat_feat).sum(1) == 4
        # expanded_mask = seat_equal_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape: [B, 1, 1, 1]
        # expanded_mask = expanded_mask.expand_as(public_card_feat)  # Shape: [B, 2, 4, 15]
        # public_card_feat = public_card_feat * expanded_mask.float()# Zero out public_card_feat where mask is False



        #多次动作组成的一轮数据，将多次打平成第三维 
        b, a, c, d, e, f = play_card_history_feat.shape
        play_card_history_feat = play_card_history_feat.reshape(b, a*c, d, e, f)
        b, a, c, d = play_seat_history_feat.shape
        play_seat_history_feat = play_seat_history_feat.reshape(b, a*c, d)
        b, a, c, d, e = bid_card_history_feat.shape

        bid_card_history_feat = bid_card_history_feat.reshape(b, a, c, d, e)
        b, a, c = bid_seat_history_feat.shape
        bid_seat_history_feat = bid_seat_history_feat.reshape(b, a, c)
        b, a, c, d, e = play_card_round_feat.shape
        play_card_round_feat = play_card_round_feat.reshape(b, a, c, d, e)
        b, a, c = play_seat_round_feat.shape
        play_seat_round_feat = play_seat_round_feat.reshape(b, a, c)
        
        card_feature_dict = {
            'play_history': play_card_history_feat.reshape(-1, 2, 4, 15),
            'bid_history': bid_card_history_feat.reshape(-1, 2, 4, 15),
            'round_history': play_card_round_feat.reshape(-1, 2, 4, 15),
            'played_history': played_card_history_feat,
            'score_remain': score_remain_card_feat,
            # 'public_card': public_card_feat,
            'mask_card': mask_cards,
        }

        
        # Concatenate all features along batch dimension
        batched_features = t.cat(list(card_feature_dict.values()), dim=0)

        # Single encoder call
        encoded_features = self.card_encoder(batched_features)

        # Split and reshape back
        split_sizes = [v.shape[0] for v in card_feature_dict.values()]
        split_features = t.split(encoded_features, split_sizes, dim=0)

        # Reconstruct individual features
        features_iter = iter(split_features)
        play_card_history_enocdefeat = next(features_iter).reshape( play_card_history_feat.shape[0], play_card_history_feat.shape[1], -1)
        bid_card_history_encodefeat = next(features_iter).reshape( bid_card_history_feat.shape[0], bid_card_history_feat.shape[1], -1)
        round_card_history_encodefeat = next(features_iter).reshape( play_card_round_feat.shape[0], play_card_round_feat.shape[1], -1)
        played_card_history_encodefeat = next(features_iter)
        # score_card_encodefeat = next(features_iter)
        score_remain_card_encodefeat = next(features_iter)
        # public_card_encodefeat = next(features_iter)
        mask_card_encodefeat = next(features_iter)

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
        play_history_feat = t.cat([play_card_history_enocdefeat, play_seat_history_feat], dim=-1)
        h_play, (_, _) = self.lstm(play_history_feat)
        # 使用注意力机制替代直接取最后一个时间步
        h_play_att = self.attention_play(play_history_feat) # [B, hidden_dim*2] -> [B, hidden_dim*2]
        h_play += h_play_att

        # bid_card_history_encodefeat = self.card_encoder(bid_card_history_feat)
        bid_history_feat = t.cat([bid_card_history_encodefeat, bid_seat_history_feat], dim=-1)
        h_bid, (_, _) = self.lstm(bid_history_feat)
        # 使用注意力机制
        h_bid_att = self.attention_bid(bid_history_feat) # [B, hidden_dim*2]
        h_bid += h_bid_att

        # round_card_history_encodefeat = self.card_encoder(play_card_round_feat)
        round_history_feat = t.cat([round_card_history_encodefeat, play_seat_round_feat], dim=-1)
        h_round, (_, _) = self.lstm(round_history_feat)
        # 使用注意力机制
        h_round_att = self.attention_round(round_history_feat) # [B, hidden_dim*2]
        h_round += h_round_att

        #挨个对扑克二维特征进行提取
        # played_card_history_encodefeat = self.card_encoder(played_card_history_feat)
        # score_card_encodefeat = self.card_encoder(score_card_feat)
        # score_remain_card_encodefeat = self.card_encoder(score_remain_card_feat)
        # public_card_encodefeat = self.card_encoder(public_card_feat)        
        # mask_card_encodefeat = self.card_encoder(mask_card)

        #特征融合
        h_play = h_play.reshape(B, -1)
        h_bid = h_bid.reshape(B, -1)
        h_round = h_round.reshape(B, -1)
        in_feat = t.cat([h_play, h_bid, h_round, 
                             played_card_history_encodefeat, score_remain_card_encodefeat, 
                             mask_card_encodefeat, my_seat_feat, banker_seat_feat], dim=-1)
        
        actionType = self.actionTypeModel.select_action(in_feat)

        # #这两个加起来也就12ms左右
        # opp_logits = t.stack([head(in_feat) for head in self.opp_heads], dim=0)  # [num_opps, B, CARD_COUNT]#预测4个玩家
        # opp_logits.transpose_(1,0)

        # #对超过手牌数的回归值上线进行截断
        # # if isTrain:
        # #     player_hand_card_num = player_hand_card_num.unsqueeze(-1).expand(-1, -1, opp_logits.shape[-1])
        # #     opp_logits.clamp_(max=player_hand_card_num)
            

        # bottom_logits = self.bottom_head(in_feat)
        # bottom_logits.squeeze_(1)
        
        # return opp_logits, bottom_logits


    def toDevice(self, device):
        self._device = device
        self.to(device)
        self.card_encoder.toDevice(device)
        self.action_type_net.to(device)
        self.action_q_net.to(device)

    #计算模型大小
    def calc_model_size(self):
        total_predictmodel_params = sum(p.numel() for p in self.card_encoder.parameters())
        print("card_encoder parameters:", total_predictmodel_params)

    # def load_checkpoint(self):
    #     pass
    
    
class IdlerModel(nn.Module):
    def __init__(self, num_action_type, device):
        super().__init__()
        self._device = device
        self.num_action_type = num_action_type
        self.cards_shape = (2, 4, 15)
        hidden_dim = 512

        # 手牌特征提取器 (2,4,15) -> (hidden_channels,2,4,5) 
        ##kernelsize=3, padding=1, stride=1以保存卷积后的尺寸不变化
        self.cards_encoder = ResNet(ResidualBlock, layers = [2,2,2,2 ], hidden_channels=[14,28,56,112], \
                                        in_channels=2,out_dim=hidden_dim, kernel_size=3, padding=1, stride=1)

        
        #历史出牌时序特征, 接着 cards_encoder 输出的out_channels*4*15维出牌特征 + 4维座位号特征
        # self.lstm = nn.LSTM(hidden_dim+4, hidden_dim, batch_first=True)
        # self.attention_play = AttentionLayer( hidden_dim+4)
        # self.attention_bid = AttentionLayer( hidden_dim+4)
        # self.attention_round = AttentionLayer( hidden_dim+4)

        #上层策略网络
        self.strategy_net = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 7, bias=False),
            nn.Sigmoid()
        )
        #下层出牌牌型网络
        self.action_type_net = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, num_action_type, bias=False),
        )
        #下层出牌Q值网络
        self.action_q_net = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 3, bias=False),
            nn.Sigmoid()
        )
        
        
        # #牌型决策层模型， 输出4种牌型概率
        # self.actionTypeModel = PPOClip(3620, 0, self.action_type_num, device=device)
        # #具体出牌层模型，输出牌型中每个具体牌的回归值
        # self.actionCardModel = PPOClip(3620, self.action_type_num, 1, device=device)
        # #垫牌模层模型，输出需要垫的牌矩阵
        # self.actionDiscardModel = PPOClip(3620, self.action_type_num+2*4*15, 2*4*15, device=device)

    def __is_tractor_batch(self, cnt):
        # cnt: [B,4,15]
        B = cnt.size(0)

        # 1. 用到的花色数量
        suit_used = (cnt.sum(dim=2) > 0).int()     # [B,4]
        suit_count = suit_used.sum(dim=1)          # [B]

        cond_one_suit = (suit_count == 1)

        # 2. 找出那个花色
        suit_idx = suit_used.argmax(dim=1)         # [B]

        # 3. 取该花色的 rank 计数
        batch_idx = t.arange(B, device=cnt.device)
        rank_cnt = cnt[batch_idx, suit_idx]        # [B,15]

        # 4. 必须全是 2（不能有 1）
        cond_all_pairs = ((rank_cnt == 2) | (rank_cnt == 0)).all(dim=1)

        # 5. 至少两个对子
        pair_mask = (rank_cnt == 2)
        pair_num = pair_mask.sum(dim=1)
        cond_min_len = (pair_num >= 2)

        # 6. 连续性判断
        # 取出 rank 下标
        is_continuous = []
        for b in range(B):
            ranks = t.where(pair_mask[b])[0]
            if len(ranks) >= 2 and t.all(ranks[1:] - ranks[:-1] == 1):
                is_continuous.append(True)
            else:
                is_continuous.append(False)

        cond_continuous = t.tensor(is_continuous, device=cnt.device)

        return cond_one_suit & cond_all_pairs & cond_min_len & cond_continuous

    def forward_tp(self, z, x, mask, return_value=False, flags=None):
        x_feat = self.cards_encoder(x)
        #连接x与z，送入action_type_net
        x_feat = x_feat.flatten(1, 2)
        z_expanded = z.unsqueeze(0).expand(self.num_action_type, -1, -1)
        t_feat = [z_expanded[i] for i in range(self.num_action_type)] + [x_feat]
        output = t.cat(t_feat, dim=-1)
        
        logits = self.action_type_net(output)
        out = logits * mask
        
        if return_value:
            return dict(values=out)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = t.randint(out.shape[0], (1,))[0]
            else:
                action = t.argmax(out, dim=0)[0]
            return dict(action=action, max_value=t.max(out), values=out)
        
    
    def forward_act(self, z, x, return_value=False, flags=None):
        x_feat = self.cards_encoder(x)
        output = t.cat([z, z, z, z, x_feat], dim=-1)
        
        logits = self.action_q_net(output)
        
        win_rate, win, lose = t.split(logits, (1, 1, 1), dim=-1)
        win_rate = t.tanh(win_rate)
        _win_rate = (win_rate + 1) / 2
        out = _win_rate * win + (1. - _win_rate) * lose

        if return_value:
            return dict(values=(win_rate, win, lose))
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = t.randint(out.shape[0], (1,))[0]
            else:
                action = t.argmax(out, dim=0)[0]
            return dict(action=action, max_value=t.max(out), values=out)
        
    
    def forward(self, obs, play_action_seq, legal_actions):
        cards_feat = self.cards_encoder(obs)
        
        play_seq_feat = self.cards_encoder(play_action_seq)
        h_play, (_, _) = self.lstm(play_seq_feat)
        # 使用注意力机制替代直接取最后一个时间步
        h_play_att = self.attention_play(play_seq_feat) # [B, hidden_dim*2] -> [B, hidden_dim*2]
        h_play += h_play_att
        
        pub_x = t.cat([cards_feat, h_play], dim=0)
        
        strategy_logits = self.strategy_net(pub_x)
        best_s = t.argmax(strategy_logits, dim=1)
        strategy_feat = t.zeros((7,), device=self._device, dtype=t.float32)
        strategy_feat[best_s] = 1.
        
        input_ty = t.cat([pub_x, strategy_feat], dim=0)
        action_type_logits = self.action_type_net(input_ty)
        best_t = t.argmax(action_type_logits, dim=1)
        
        actions_batch = t.tensor(legal_actions[best_t], dtype=t.float32, device=self._device)
        pub_x_batch = np.repeat( pub_x[np.newaxis, :, :], actions_batch.shape[0], axis=0 )
        strategy_feat_batch = np.repeat( strategy_feat[np.newaxis, :, :], actions_batch.shape[0], axis=0 )
        z_batch = np.concatenate((actions_batch, pub_x_batch, strategy_feat_batch), axis=1)
        act_logits = self.action_q_net(z_batch)
        
        return dict(value=act_logits, strategy=best_s)
    def forward1(self, state_feat, isTrain = None):
        B = state_feat['history_play_seat'].shape[0]#批次
        play_card_history_feat = state_feat['history_play_card']#历史出牌
        play_seat_history_feat = state_feat['history_play_seat']#历史出牌座位号
        played_card_history_feat=state_feat['history_played_card']#已出过的牌
        bid_card_history_feat = state_feat['history_bid_card']#報主記錄
        bid_seat_history_feat = state_feat['history_bid_seat']#報主座位号
        
        play_card_round_feat = state_feat['round_play_card']#当前回合牌
        play_seat_round_feat = state_feat['round_play_seat']#当前回合牌

        score_card_feat = state_feat['score_card']#分數牌
        score_remain_card_feat = state_feat['remain_score_card']#分數牌
        my_seat_feat = state_feat['my_seat']#我的座位号
        banker_seat_feat = state_feat['banker_seat']#我的座位号
        mask_cards = state_feat['mask_card']#.copy()#mask牌
        hand_card_feat = state_feat['hand_card']#.copy()
        public_card_feat = state_feat['public_card']
        legal_actions = state_feat['legal_actions']#合法动作，【单牌，对子，甩牌，拖拉机】4维特征，对应
        action_types = np.array(list(legal_actions.keys()))
        action_cards = list(legal_actions.values())#合法
        
        # #是否首出
        # player0_cards = play_card_round_feat[:,0]#[B,4,2,4,15] -> [B,2,4,15]
        # cnt = player0_cards.sum(dim=1)   # [B,4,15] cnt[b, s, r] ∈ {0,1,2} 第 b 个 batch 中，玩家 0 在 (suit=s, rank=r) 上有几张
        # total_cards = cnt.sum(dim=(1,2))   # [B] 总牌数
        # is_first_play = (total_cards == 0)#首出
        # is_single = (total_cards == 1)
        # is_pair = (total_cards == 2) & (cnt.max(dim=2).values.max(dim=1).values == 2)
        # tractor_mask = self.__is_tractor_batch(cnt)
        # round_play_card_type = t.full((B,), fill_value=-1, device=cnt.device)
        # round_play_card_type[is_single] = 0#单张
        # round_play_card_type[is_pair] = 1#对子
        # round_play_card_type[tractor_mask] = 2#拖拉机
        # round_play_card_type[round_play_card_type == -1] = 3# 剩下的全部是甩牌



        
        ################################################################################

        # #底牌，只有庄家知道
        # public_card_feat = state_feat['public_card']
        # seat_equal_mask = (my_seat_feat == banker_seat_feat).sum(1) == 4
        # expanded_mask = seat_equal_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape: [B, 1, 1, 1]
        # expanded_mask = expanded_mask.expand_as(public_card_feat)  # Shape: [B, 2, 4, 15]
        # public_card_feat = public_card_feat * expanded_mask.float()# Zero out public_card_feat where mask is False



        #多次动作组成的一轮数据，将多次打平成第三维 
        b, a, c, d, e, f = play_card_history_feat.shape
        play_card_history_feat = play_card_history_feat.reshape(b, a*c, d, e, f)
        b, a, c, d = play_seat_history_feat.shape
        play_seat_history_feat = play_seat_history_feat.reshape(b, a*c, d)
        b, a, c, d, e = bid_card_history_feat.shape

        bid_card_history_feat = bid_card_history_feat.reshape(b, a, c, d, e)
        b, a, c = bid_seat_history_feat.shape
        bid_seat_history_feat = bid_seat_history_feat.reshape(b, a, c)
        b, a, c, d, e = play_card_round_feat.shape
        play_card_round_feat = play_card_round_feat.reshape(b, a, c, d, e)
        b, a, c = play_seat_round_feat.shape
        play_seat_round_feat = play_seat_round_feat.reshape(b, a, c)
        
        card_feature_dict = {
            'play_history': play_card_history_feat.reshape(-1, 2, 4, 15),
            'bid_history': bid_card_history_feat.reshape(-1, 2, 4, 15),
            'round_history': play_card_round_feat.reshape(-1, 2, 4, 15),
            'played_history': played_card_history_feat,
            'score_remain': score_remain_card_feat,
            # 'public_card': public_card_feat,
            'mask_card': mask_cards,
        }

        
        # Concatenate all features along batch dimension
        batched_features = t.cat(list(card_feature_dict.values()), dim=0)

        # Single encoder call
        encoded_features = self.card_encoder(batched_features)

        # Split and reshape back
        split_sizes = [v.shape[0] for v in card_feature_dict.values()]
        split_features = t.split(encoded_features, split_sizes, dim=0)

        # Reconstruct individual features
        features_iter = iter(split_features)
        play_card_history_enocdefeat = next(features_iter).reshape( play_card_history_feat.shape[0], play_card_history_feat.shape[1], -1)
        bid_card_history_encodefeat = next(features_iter).reshape( bid_card_history_feat.shape[0], bid_card_history_feat.shape[1], -1)
        round_card_history_encodefeat = next(features_iter).reshape( play_card_round_feat.shape[0], play_card_round_feat.shape[1], -1)
        played_card_history_encodefeat = next(features_iter)
        # score_card_encodefeat = next(features_iter)
        score_remain_card_encodefeat = next(features_iter)
        # public_card_encodefeat = next(features_iter)
        mask_card_encodefeat = next(features_iter)

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
        play_history_feat = t.cat([play_card_history_enocdefeat, play_seat_history_feat], dim=-1)
        h_play, (_, _) = self.lstm(play_history_feat)
        # 使用注意力机制替代直接取最后一个时间步
        h_play_att = self.attention_play(play_history_feat) # [B, hidden_dim*2] -> [B, hidden_dim*2]
        h_play += h_play_att

        # bid_card_history_encodefeat = self.card_encoder(bid_card_history_feat)
        bid_history_feat = t.cat([bid_card_history_encodefeat, bid_seat_history_feat], dim=-1)
        h_bid, (_, _) = self.lstm(bid_history_feat)
        # 使用注意力机制
        h_bid_att = self.attention_bid(bid_history_feat) # [B, hidden_dim*2]
        h_bid += h_bid_att

        # round_card_history_encodefeat = self.card_encoder(play_card_round_feat)
        round_history_feat = t.cat([round_card_history_encodefeat, play_seat_round_feat], dim=-1)
        h_round, (_, _) = self.lstm(round_history_feat)
        # 使用注意力机制
        h_round_att = self.attention_round(round_history_feat) # [B, hidden_dim*2]
        h_round += h_round_att

        #挨个对扑克二维特征进行提取
        # played_card_history_encodefeat = self.card_encoder(played_card_history_feat)
        # score_card_encodefeat = self.card_encoder(score_card_feat)
        # score_remain_card_encodefeat = self.card_encoder(score_remain_card_feat)
        # public_card_encodefeat = self.card_encoder(public_card_feat)        
        # mask_card_encodefeat = self.card_encoder(mask_card)

        #特征融合
        h_play = h_play.reshape(B, -1)
        h_bid = h_bid.reshape(B, -1)
        h_round = h_round.reshape(B, -1)
        in_feat = t.cat([h_play, h_bid, h_round, 
                             played_card_history_encodefeat, score_remain_card_encodefeat, 
                             mask_card_encodefeat, my_seat_feat, banker_seat_feat], dim=-1)
        
        actionType = self.actionTypeModel.select_action(in_feat)

        # #这两个加起来也就12ms左右
        # opp_logits = t.stack([head(in_feat) for head in self.opp_heads], dim=0)  # [num_opps, B, CARD_COUNT]#预测4个玩家
        # opp_logits.transpose_(1,0)

        # #对超过手牌数的回归值上线进行截断
        # # if isTrain:
        # #     player_hand_card_num = player_hand_card_num.unsqueeze(-1).expand(-1, -1, opp_logits.shape[-1])
        # #     opp_logits.clamp_(max=player_hand_card_num)
            

        # bottom_logits = self.bottom_head(in_feat)
        # bottom_logits.squeeze_(1)
        
        # return opp_logits, bottom_logits


    def toDevice(self, device):
        self._device = device
        self.to(device)
        self.card_encoder.toDevice(device)
        self.action_type_net.to(device)
        self.action_q_net.to(device)

    #计算模型大小
    def calc_model_size(self):
        total_predictmodel_params = sum(p.numel() for p in self.card_encoder.parameters())
        print("card_encoder parameters:", total_predictmodel_params)

    # def load_checkpoint(self):
    #     pass