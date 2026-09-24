import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import os
import datetime
from .BasicBlockM import ResNet, ResidualBlock, ResidualLinearBlock
# from .common_model import PlayerEncoder

# 上层决策：7 个可学习的离散策略（option），由整局回报优化。
NUM_STRATEGY = 7

# 事后动作诊断标签；索引不约束回报学习后的 option 语义。
_STRATEGY_NAMES = ['强势进攻', '稳健控分', '同伴助攻', '破坏防守', '垫牌过渡', '保底策略', '甩牌试探']

# 庄家方阵营；闲家方为其余两个角色
_BANKER_TEAM = {'banker', 'banker_op'}


def compute_strategy_targets(step_meta, position):
    """由「本墩实际动作特征」计算 7 维 one-hot 诊断标签（互斥单标签）。

    这些标签使用出牌后的结果，只用于监控；不是上层 Q(s, g) 的回归目标。

    step_meta: Env._get_step_meta() 在环境 reset 之前冻结的本墩快照
        roles[role] = {is_lead, threw, followed, won, score_in_play}
        winner_role / round_score / score_played / is_last_round

    优先级自上而下，命中一条即返回，天然 one-hot、不会出现并列：
        5 保底策略   终局墩且我方（自己或队友）夺得本墩牌权
        6 甩牌试探   首出且实际甩牌（牌型判定为 SUSPECT）
        4 垫牌过渡   非首出且自己未赢墩，包含跟门和未跟门的牌
        2 同伴助攻   队友夺得本墩牌权且本墩打出过分牌
        1 稳健控分   自己夺得牌权且本墩打出过分牌
        0 强势进攻   自己夺得牌权
        3 破坏防守   其余（对手夺得牌权）兜底
    """
    one_hot = np.zeros(NUM_STRATEGY, dtype=np.float32)
    step_meta = step_meta or {}
    feat = step_meta.get('roles', {}).get(position)
    if feat is None:  # 缺少本墩信息时退化为强势进攻
        one_hot[0] = 1.0
        return one_hot

    winner_role = step_meta.get('winner_role')
    if winner_role is None:
        mate_won = False
    else:
        same_team = (winner_role in _BANKER_TEAM) == (position in _BANKER_TEAM)
        mate_won = same_team and winner_role != position

    is_last = float(step_meta.get('is_last_round', 0.0)) >= 0.5
    score_played = float(step_meta.get('score_played', 0.0)) >= 0.5
    is_lead = feat['is_lead'] >= 0.5
    won = feat['won'] >= 0.5
    
    idx = 3
    if is_lead:
        if won:
            if is_last:
                idx = 5#5 保底策略   终局墩且我方（自己或队友）夺得本墩牌权
            if score_played:
                idx = 1#1 稳健控分   自己夺得牌权且本墩打出过分牌
            else:
                idx = 0#0 强势进攻   自己夺得牌权
        elif feat['threw'] >= 0.5:
            idx = 6#6 甩牌试探   首出且实际甩牌（牌型判定为 SUSPECT）
        elif not score_played:
            idx = 4#4 垫牌过渡   自己未赢墩，包含跟门和未跟门的牌
    else:
        if won:
            if is_last:
                idx = 5#5 保底策略   终局墩且我方（自己或队友）夺得本墩牌权
            elif score_played:
                idx = 1#1 稳健控分   自己夺得牌权且本墩打出过分牌
            else:
                idx = 0#0 强势进攻   自己夺得牌权
        elif mate_won:
            if is_last:
                idx = 5#5 保底策略   终局墩且我方（自己或队友）夺得本墩牌权
            elif score_played:
                idx =2#2 同伴助攻   队友夺得本墩牌权且本墩打出过分牌
            elif not score_played:
                idx = 4#4 垫牌过渡   自己未赢墩，包含跟门和未跟门的牌
        else:
            if not score_played:
                idx = 4#4 垫牌过渡   非首出且自己未赢墩，包含跟门和未跟门的牌

    one_hot[idx] = 1.0
    return one_hot

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
        hidden_dim = 4096

        # 手牌特征提取器 (2,4,15) -> (hidden_channels,2,4,5) 
        ##kernelsize=3, padding=1, stride=1以保存卷积后的尺寸不变化
        self.cards_encoder_tp = ResNet(ResidualBlock, layers = [2,2,2,2 ], hidden_channels=[58,58,116,116], \
                                        in_channels=58,out_dim=hidden_dim, kernel_size=3, padding=1, stride=1)

        self.cards_encoder_action = ResNet(ResidualBlock, layers = [2,2,2,2 ], hidden_channels=[60,60,120,120], \
                                        in_channels=60,out_dim=hidden_dim, kernel_size=3, padding=1, stride=1)
        #历史出牌时序特征, 接着 cards_encoder 输出的out_channels*4*15维出牌特征 + 4维座位号特征
        # self.lstm = nn.LSTM(hidden_dim+4, hidden_dim, batch_first=True)
        # self.attention_play = AttentionLayer( hidden_dim+4)
        # self.attention_bid = AttentionLayer( hidden_dim+4)
        # self.attention_round = AttentionLayer( hidden_dim+4)

        #上层策略网络：状态级输入(不含牌型候选)，z(297) + x_feat(4096)
        self.strategy_net = nn.Sequential(
            ResidualLinearBlock(4393, 512),
            ResidualLinearBlock(512, 128),
            nn.Linear(128, NUM_STRATEGY, bias=False),
        )

        #下层牌型网络
        self.type_q_net = nn.Sequential(
            ResidualLinearBlock(5005 + NUM_STRATEGY*3, 2048),
            ResidualLinearBlock(2048, 512),
            ResidualLinearBlock(512, 128),
            nn.Linear(128, 3, bias=False),
        )
        #下层出牌网络
        self.action_q_net = nn.Sequential(
            ResidualLinearBlock(5005 + NUM_STRATEGY*3, 2048),
            ResidualLinearBlock(2048, 512),
            ResidualLinearBlock(512, 128),
            nn.Linear(128, 3, bias=False),
        )
        
        # #牌型决策层模型， 输出4种牌型概率
        # self.actionTypeModel = PPOClip(3620, 0, self.action_type_num, device=device)
        # #具体出牌层模型，输出牌型中每个具体牌的回归值
        # self.actionCardModel = PPOClip(3620, self.action_type_num, 1, device=device)
        # #垫牌模层模型，输出需要垫的牌矩阵
        # self.actionDiscardModel = PPOClip(3620, self.action_type_num+2*4*15, 2*4*15, device=device)

    def forward_strategy(self, z, x, return_value=False, flags=None):
        """输出 Q(s, g)；单状态推理按 epsilon-greedy 选择上层策略。"""
        x_feat = self.cards_encoder_tp(x)
        output = t.cat([z, x_feat], -1)
        values = self.strategy_net(output)  # [B, NUM_STRATEGY]
        
        if return_value:
            return dict(values=values)

        if values.shape[0] != 1:
            raise ValueError('Strategy selection requires exactly one state')
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            strategy = t.randint(NUM_STRATEGY, (), device=values.device)
        else:
            strategy = values[0].argmax(dim=-1)
        return dict(action=strategy, values=values)

    
    def forward_tp(self, z, x, strategy, return_value=False, flags=None):
        x_feat = self.cards_encoder_tp(x)
        output = t.cat([z, z, z, strategy, strategy, strategy, x_feat], -1)
        
        logits = self.type_q_net(output)
        win_rate, win, lose = t.split(logits, (1, 1, 1), dim=-1)
        # win/lose 同样 tanh 有界化到 [-1,1]：无界线性输出与 target_adp 的 MSE 会因单批极端输出产生巨大 loss 尖峰
        # win_rate = t.tanh(win_rate)
        # win = t.tanh(win)
        # lose = t.tanh(lose)
        _win_rate = t.sigmoid(win_rate)  # 训练回归 1{wp>0}（BCEWithLogits），输出是胜率 logit，sigmoid 才是概率
        out = _win_rate * win + (1. - _win_rate) * lose
        
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = t.randint(out.shape[0], (1,))[0]
        else:
            action = t.argmax(out, dim=0)[0]
                
        if return_value:
            return dict(action=action, max_value=t.max(out), values=(win_rate, win, lose))
        else:
            return dict(action=action, max_value=t.max(out))

    def forward_act(self, z, x, strategy, return_value=False, flags=None):
        x_feat = self.cards_encoder_action(x)
        output = t.cat([z, z, z, strategy, strategy, strategy, x_feat], -1)

        logits = self.action_q_net(output)
        win_rate, win, lose = t.split(logits, (1, 1, 1), dim=-1)
        # win/lose 同样 tanh 有界化到 [-1,1]：无界线性输出与 target_adp 的 MSE 会因单批极端输出产生巨大 loss 尖峰
        # win_rate = t.tanh(win_rate)
        # win = t.tanh(win)
        # lose = t.tanh(lose)
        _win_rate = t.sigmoid(win_rate)  # 训练回归 1{wp>0}（BCEWithLogits），输出是胜率 logit，sigmoid 才是概率
        out = _win_rate * win + (1. - _win_rate) * lose

        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = t.randint(out.shape[0], (1,))[0]
        else:
            action = t.argmax(out, dim=0)[0]
                
        if return_value:
            return dict(action=action, max_value=t.max(out), values=(win_rate, win, lose))
        else:
            return dict(action=action, max_value=t.max(out))
        
    
    def toDevice(self, device):
        self._device = device
        self.to(device)
        self.cards_encoder_tp.to(device)
        self.cards_encoder_action.to(device)
        self.strategy_net.to(device)
        self.type_q_net.to(device)
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
        hidden_dim = 4096
        #hidden_dim = 4096 hidden_channels=[58,116,232,464], ResidualLinearBlock(5005, 2048),

        # 手牌特征提取器 (2,4,15) -> (hidden_channels,2,4,5) 
        ##kernelsize=3, padding=1, stride=1以保存卷积后的尺寸不变化
        self.cards_encoder_tp = ResNet(ResidualBlock, layers = [2,2,2,2 ], hidden_channels=[58,58,116,116], \
                                        in_channels=58,out_dim=hidden_dim, kernel_size=3, padding=1, stride=1)

        self.cards_encoder_action = ResNet(ResidualBlock, layers = [2,2,2,2 ], hidden_channels=[60,60,120,120], \
                                        in_channels=60,out_dim=hidden_dim, kernel_size=3, padding=1, stride=1)
        #历史出牌时序特征, 接着 cards_encoder 输出的out_channels*4*15维出牌特征 + 4维座位号特征
        # self.lstm = nn.LSTM(hidden_dim+4, hidden_dim, batch_first=True)
        # self.attention_play = AttentionLayer( hidden_dim+4)
        # self.attention_bid = AttentionLayer( hidden_dim+4)
        # self.attention_round = AttentionLayer( hidden_dim+4)

        #上层策略网络：状态级输入(不含牌型候选)，z(297) + x_feat(4096)
        self.strategy_net = nn.Sequential(
            ResidualLinearBlock(4393, 512),
            ResidualLinearBlock(512, 128),
            nn.Linear(128, NUM_STRATEGY, bias=False),
        )

        #下层牌型网络
        self.type_q_net = nn.Sequential(
            ResidualLinearBlock(5005 + NUM_STRATEGY*3, 2048),
            ResidualLinearBlock(2048, 512),
            ResidualLinearBlock(512, 128),
            nn.Linear(128, 3, bias=False),
        )
        #下层出牌网络
        self.action_q_net = nn.Sequential(
            ResidualLinearBlock(5005 + NUM_STRATEGY*3, 2048),
            ResidualLinearBlock(2048, 512),
            ResidualLinearBlock(512, 128),
            nn.Linear(128, 3, bias=False),
        )
        
        # #牌型决策层模型， 输出4种牌型概率
        # self.actionTypeModel = PPOClip(3620, 0, self.action_type_num, device=device)
        # #具体出牌层模型，输出牌型中每个具体牌的回归值
        # self.actionCardModel = PPOClip(3620, self.action_type_num, 1, device=device)
        # #垫牌模层模型，输出需要垫的牌矩阵
        # self.actionDiscardModel = PPOClip(3620, self.action_type_num+2*4*15, 2*4*15, device=device)

    def forward_strategy(self, z, x, return_value=False, flags=None):
        """输出 Q(s, g)；单状态推理按 epsilon-greedy 选择上层策略。"""
        x_feat = self.cards_encoder_tp(x)
        output = t.cat([z, x_feat], -1)
        values = self.strategy_net(output)  # [B, NUM_STRATEGY]，不是分类 logits

        if return_value:
            return dict(values=values)

        if values.shape[0] != 1:
            raise ValueError('Strategy selection requires exactly one state')
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            strategy = t.randint(NUM_STRATEGY, (), device=values.device)
        else:
            strategy = values[0].argmax(dim=-1)
        return dict(action=strategy, values=values)


    def forward_tp(self, z, x, strategy, return_value=False, flags=None):
        x_feat = self.cards_encoder_tp(x)
        output = t.cat([z, z, z, strategy, strategy, strategy, x_feat], -1)
        
        logits = self.type_q_net(output)
        win_rate, win, lose = t.split(logits, (1, 1, 1), dim=-1)
        # win/lose 同样 tanh 有界化到 [-1,1]：无界线性输出与 target_adp 的 MSE 会因单批极端输出产生巨大 loss 尖峰
        # win_rate = t.tanh(win_rate)
        # win = t.tanh(win)
        # lose = t.tanh(lose)
        _win_rate = t.sigmoid(win_rate)  # 训练回归 1{wp>0}（BCEWithLogits），输出是胜率 logit，sigmoid 才是概率
        out = _win_rate * win + (1. - _win_rate) * lose
        
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = t.randint(out.shape[0], (1,))[0]
        else:
            action = t.argmax(out, dim=0)[0]
                
        if return_value:
            return dict(action=action, max_value=t.max(out), values=(win_rate, win, lose))
        else:
            return dict(action=action, max_value=t.max(out))

    def forward_act(self, z, x, strategy, return_value=False, flags=None):
        x_feat = self.cards_encoder_action(x)
        output = t.cat([z, z, z, strategy, strategy, strategy, x_feat], -1)
        
        logits = self.action_q_net(output)        
        win_rate, win, lose = t.split(logits, (1, 1, 1), dim=-1)
        # win/lose 同样 tanh 有界化到 [-1,1]：无界线性输出与 target_adp 的 MSE 会因单批极端输出产生巨大 loss 尖峰
        # win_rate = t.tanh(win_rate)
        # win = t.tanh(win)
        # lose = t.tanh(lose)
        _win_rate = t.sigmoid(win_rate)  # 训练回归 1{wp>0}（BCEWithLogits），输出是胜率 logit，sigmoid 才是概率
        out = _win_rate * win + (1. - _win_rate) * lose

        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action = t.randint(out.shape[0], (1,))[0]
        else:
            action = t.argmax(out, dim=0)[0]
                
        if return_value:
            return dict(action=action, max_value=t.max(out), values=(win_rate, win, lose))
        else:
            return dict(action=action, max_value=t.max(out))
        
    
    def toDevice(self, device):
        self._device = device
        self.to(device)
        self.cards_encoder_tp.to(device)
        self.cards_encoder_action.to(device)
        self.strategy_net.to(device)
        self.type_q_net.to(device)
        self.action_q_net.to(device)

    #计算模型大小
    def calc_model_size(self):
        total_predictmodel_params = sum(p.numel() for p in self.card_encoder.parameters())
        print("card_encoder parameters:", total_predictmodel_params)

    # def load_checkpoint(self):
    #     pass
