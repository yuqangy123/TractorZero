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
import torch.nn as nn
from rlcard.games.tractors.models.stractor_resnet import ResNet, ResidualBlock

'''出牌模型'''

#玩家出牌信息编码器
class PlayerEncoder(nn.Module):
    """
    玩家出牌信息特征编码器，用于表示历史出牌中或当前回合中玩家的出牌信息特征
    由手牌+座位号+阵营组成，座位号需转为以自己为1号位的本地座位号
    hand_matrix: [batch, 2, 14, 4] 手牌矩阵
    seat_id: [batch] 座位ID (1-4)
    team_id: [batch] 阵营ID (1或2，1为我方，2为敌方)
    经过融合层输出128维的特征向量
    """
    def __init__(self):
        super(PlayerEncoder, self).__init__()
        cards_embed_dim=2*14*4
        seat_embed_dim=4
        team_embed_dim=2
        self.output_dim = 128
        
        # 手牌特征提取器 (2,14,4) -> 256维
        self.cards_restnet = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=3)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(cards_embed_dim + seat_embed_dim + team_embed_dim, (cards_embed_dim + seat_embed_dim + team_embed_dim)*2),
            nn.LeakyReLU(),
            nn.Linear((cards_embed_dim + seat_embed_dim + team_embed_dim)*2, self.output_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(self.output_dim))
    
    def forward(self, hand_matrix, seat_emb, team_emb):
        b = hand_matrix.shape[0]#batch
        if b == 0:
            return t.zeros(0, self.output_dim)

        # 提取手牌特征
        hand_card_feat = self.cards_restnet(hand_matrix)
        
        # 位置和阵营嵌入属性
        # seat_emb = t.zeros((b,4), device=seat_id.device)
        # # 使用 scatter_ 在 dim=1 上，按 seat_id-1 的索引赋值 1
        # seat_emb.scatter_(1, (seat_id - 1).long(), 1)
        # team_emb = t.zeros((b,2), device=team_id.device)
        # team_emb.scatter_(1, (team_id - 1).long(), 1)
        
        # 融合特征
        combined = t.cat([hand_card_feat, seat_emb, team_emb], dim=1)
        player_embed = self.fusion(combined)
        
        return player_embed



class Actor(nn.Module):
    """
    通过状态编码器编码场面信息。
    出牌一般由固定牌+垫牌组成，固定牌为必须出的牌，垫牌为可以随意出的牌。例如在节拖拉机的牌时，若花色中只有一对牌和若干单牌，则一对牌为固定牌，另外还需要从剩余的牌中取出两张单牌，此为垫牌
    动作预测的输入是状态编码+N种固定牌组合+可垫牌，输出是预测每张可垫牌的概率
    """

    def __init__(self) -> None:
        super().__init__()

        '''
        状态编码器
        
        历史出牌编码网络，只输入最近10回合的 历史出牌
        使用resnet提取出牌特征，再输入lstm提取出牌历史时序信息。
        这个网络设计有效结合了卷积神经网络的空间特征提取能力和循环神经网络的时序建模能力，
        特别适合处理像牌局序列这样具有时间依赖性的结构化数据。通过ResNet处理每个时间步的2D特征，再通过LSTM整合时序信息，模型能够捕捉牌局间的复杂动态关系。

        
        '''        
        self.cards_restnet = ResNet(ResidualBlock, [2, 2, 2, 2], in_channels=2, kernel_size=3)
        self.player_encoder = PlayerEncoder()
        self.lstm = nn.LSTM(self.player_encoder.output_dim, 96, batch_first=True)
        
        # 状态投影层（用于点积注意力）
        self.state_fusion_net = nn.Sequential(
            nn.Linear(108, n_state_dim),
            nn.ReLU(),
            nn.Linear(n_state_dim, n_state_dim),
            nn.LayerNorm(n_state_dim)
        )


        # 固定动作编码器
        self.fixed_action_net = nn.Sequential(
            nn.Linear(108, n_state_dim),
            nn.ReLU(),
            nn.Linear(n_state_dim, n_state_dim),
            nn.LayerNorm(n_state_dim)
        )

        # 高效的注意力评分网络（双线性形式）
        self.attention = nn.Bilinear(n_state_dim, n_state_dim, 1, bias=False)
        
        

        # 垫牌动作编码器
        self.action_discard_encoder = nn.Sequential(
            nn.Linear(108, n_state_dim),
            nn.ReLU(),
            nn.Linear(n_state_dim, n_state_dim),
            nn.LayerNorm(n_state_dim)
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

        # 底牌预测模块
        self.public_card_net = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        # 底牌概率预测头
        self.pro_public_card_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Sigmoid()
        )

        # 概率预测头
        self.probability_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Sigmoid()
        )

        # 可学习的标准差 log(σ)，避免数值不稳定
        self.log_std = nn.Parameter(t.zeros(1, 2*14*4))

    def forward(self, obs_x, actions_fixed, actions_discard) -> Tensor:
        #出牌方位为调整为以自己为0号位
        play_card_history_feat = obs_x['history_play_card']#历史出牌
        play_seat_history_feat = obs_x['history_play_seat']#出牌座位号
        play_team_history_feat = obs_x['history_play_team']#出牌阵营

        play_history_feat = self.player_encoder(play_card_history_feat, play_seat_history_feat, play_team_history_feat)
        lstm_out, (h_n, _) = self.lstm(play_history_feat)
        
        own_seat = obs_x['seat']#我的座位号
        hand_cards = obs_x['hand_cards']#我的手牌
        curr_round_play_cards_feat =  self.player_encoder(obs_x['round_play_card'], obs_x['round_play_seat'], obs_x['round_play_team'])#当前轮的出牌特征
        last_play_cards_feat = self.player_encoder(play_card_history_feat[-1].unsqueeze(0), play_seat_history_feat[-1].unsqueeze(0), play_team_history_feat[-1].unsqueeze(0))#上一轮出牌
        played_cards = obs_x['played_cards'].unsqueeze(0)#已经出过的牌
        level_cards = obs_x['level_card'].unsqueeze(0)#当前打第几级，用一副扑克表示[1,14,4]当前的级牌
        score_cards = obs_x['score_card'].unsqueeze(0)#当前得分，庄家从80分算起，当闲家得到80分时得分为0，超过80分时得分为负数；闲家从-80分算起，得到80分时得分为0，超过80分时得分为正数，用一副扑克表示（5,10，k）[2,14,4]
        remain_score_cards = obs_x['remain_score_card'].unsqueeze(0)#场面剩余分数牌，用两副扑克表示（5,10，k）[2,14,4]
        combined = t.cat([played_cards, level_cards, score_cards, remain_score_cards], dim=0)
        combined_feat = self.cards_restnet(combined)#联合计算，提高速度
        
        #预测八张底牌(n/25 * pre_public_card)
        if obs_x['banker'] == own_seat:
            pre_publiccard = obs_x['public_card']
            public_card_feat = obs_x['public_card'].copy().unsqueeze(0)
        else:
            for card_mat in obs_x['round_play_card']:
                remain_cards += card_mat.deatch()
            remain_cards += hand_cards.detach()
            remain_cards = 1-remain_cards
            remain_cards[:, 2:3, :] = 0
            public_card_feat = t.cat([remain_cards, play_history_feat, curr_round_play_cards_feat], dim=0)
            public_card_feat = self.cards_restnet(public_card_feat)
            upsampled = F.interpolate(public_card_feat, size=(14, 4), mode='bilinear', align_corners=False)
            probability = self.pro_public_card_head(upsampled)
            probability *= remain_cards
        
            # distribution_publiccard = Categorical(probability)
            public_card_feat = t.multinomial(probability, num_samples=8, replacement=False)#无放回采样
            public_card_feat *= (1 - hand_cards.sum().detach()/25)#最多25张手牌
            # log_probs_publiccard = distribution_publiccard.log_prob(pre_publiccard).sum(-1)# [batch_size, k]
            # entropy_publiccard = distribution_publiccard.entropy().sum(-1)
            pre_publiccard = public_card_feat.copy()
            pre_publiccard[pre_publiccard > 0] = 1

        #融合状态特征
        fusion_feat = t.cat([hand_cards, own_seat, lstm_out, curr_round_play_cards_feat, last_play_cards_feat, combined_feat, public_card_feat], dim=1)
        state_feat = self.state_fusion_net(fusion_feat)

        #通过obs_feat场面信息+actions_fixed+actions_discard预测actions_discard中每张牌的出牌概率
        # actions_fixed_feat = self.cards_restnet(actions_fixed)
        # actions_discard_feat = self.cards_restnet(actions_discard)
        
        
        #DQN 状态广播并与固定组合动作牌融合
        state_emb = state_feat.unsqueeze(1).expand(actions_fixed.shape[0], -1)
        #预测固定组合动作牌的概率
        fixed_action_prob = self.fixed_action_net(state_emb, actions_fixed)

        discard_action_prob = t.empty(actions_discard.shape)
        #以 固定组合动作牌+状态 作为输入，预测剩余待选择牌中每张牌被选择为垫牌的概率，可采用注意力机制
        card_count = obs_x['round_play_card'][-1].sum()
        for i in range(actions_fixed.shape[0]):
            comb_feat = t.cat([state_feat, actions_fixed[i, :, :, :], actions_discard], 1) #[b, n_channels, w, h]
            comb_feat = self.cards_restnet(comb_feat)
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
    def __init__(self, args: dict):
        # select the device
        self.device = t.device(
            "cuda") if args["device"] == "cuda" and t.cuda.is_available() else "cpu"
        t.cuda.empty_cache()
        # save arguments to member variables
        self.state_dim = args["state_dim"]
        self.action_dim = args["action_dim"]
        self.hidden_dim = args["hidden_dim"]
        self.lr_actor = args["lr_actor"]
        self.lr_critic = args["lr_critic"]
        self.eps_clip = args["eps_clip"]
        self.buffer_size = args["buffer_size"]
        self.entropy_weight = args["entropy_weight"]
        self.epochs = args["epochs"]
        self.batch_mode = args["batch_mode"]
        self.batch_size = args["batch_size"]
        self.gamma = args["gamma"]
        self.lamb = args["lambda"]
        self.training_epochs = args["training_epochs"]
        self.max_episode_length = args["max_episode_length"]
        self.update_interval = args["update_interval"]
        now = datetime.datetime.now()
        folder_name = now.strftime("%Y-%m-%d %H-%M-%S")
        self.log_path = os.path.join(args["log_path"], folder_name, "logs")
        self.save_path = os.path.join(
            args["log_path"], folder_name, "saved_models")
        # make dir
        os.makedirs(self.log_path)
        os.makedirs(self.save_path)
        self.actor_path = args["actor_path"]
        self.critic_path = args["critic_path"]
        # define actor critic networks
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.hidden_dim).to(self.device)
        self.old_actor = Actor(
            self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.old_critic = Critic(
            self.state_dim, self.hidden_dim).to(self.device)
        # load the pretrained models
        if self.actor_path is not None and self.critic_path is not None:
            self.load_parameters()
        # copy parameters
        self.update_old_net()
        # optimizer for the actor and critic networks
        self.optimizer = t.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor},
                {"params": self.critic.parameters(), "lr": self.lr_critic}
            ]
        )
        # loss function for critics
        self.mseloss = nn.MSELoss()
        # rollout buffer of PPO
        self.buffer = {
            "rewards": np.zeros(self.buffer_size, dtype=np.float32),
            "dones": np.zeros(self.buffer_size, dtype=np.float32),
            "states": np.zeros((self.buffer_size, self.state_dim), dtype=np.float32),
            "actions": np.zeros(self.buffer_size, dtype=np.float32),
            "log_probs": np.zeros(self.buffer_size, dtype=np.float32),
            "values": np.zeros(self.buffer_size, dtype=np.float32),
        }
        # buffers of the derived values
        # advantage
        self.adv_buffer = np.zeros(
            self.buffer_size, dtype=np.float32)

        # discounted reward aka. return
        self.ret_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # buffer of the state variables
        self.ptr, self.path_start_idx, self.max_size = 0, 0, self.buffer_size

    def store_transition(self, reward: float, done: float, state: np.ndarray, action: float, log_prob: float, value: float):
        """
        store a transition in the buffer
        """
        assert self.ptr < self.max_size
        self.buffer["rewards"][self.ptr] = reward
        self.buffer["dones"][self.ptr] = done
        self.buffer["states"][self.ptr] = state.cpu().detach().numpy()
        self.buffer["actions"][self.ptr] = action.cpu().detach().numpy()
        self.buffer["log_probs"][self.ptr] = log_prob.cpu().detach().numpy()
        self.buffer["values"][self.ptr] = value.cpu().detach().numpy()
        # increase the pointer
        self.ptr += 1

    def finish_path(self, last_val: float = 0):
        """
        Calculate GAE(General Advantage Estimates) and discounted rewards when the game is finished.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        vals = np.append(self.buffer["values"][path_slice], last_val)
        rews = np.append(self.buffer["rewards"][path_slice], last_val)
        deltas = rews[:-1] + self.gamma*vals[1:]-vals[:-1]
        self.adv_buffer[path_slice] = self._discount_cumsum(
            deltas, self.gamma*self.lamb)
        self.ret_buffer[path_slice] = self._discount_cumsum(rews, self.gamma)[
            :-1]
        self.path_start_idx = self.ptr

    @staticmethod
    def _discount_cumsum(x, discount: float):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        """

        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def get_data(self):
        """
        sample all data from the buffers with the estimated advantage and cumulative discounted rewards
        -------
        Returns:
            the data from the buffers with GAE and returns
        """
        data = {k: self.buffer[k][:self.ptr] for k in self.buffer}
        data["advantages"] = self.adv_buffer[:self.ptr]
        data["discounted_rewards"] = self.ret_buffer[:self.ptr]
        return data

    def clean_buffer(self):
        """
        clear the buffer of past samples
        """
        # just modify the pointer and start index
        self.ptr = 0
        self.path_start_idx = 0

    # 实现“先选主动作（固定牌），再选辅动作（垫牌）”的 分层决策机制（Hierarchical Policy）
    def act(self, obs_x, actions_fixed, actions_discard, discard_num):
        # 获取两个动作的概率分布
        actions_fixed_prob, discard_action_prob, pre_publiccard = self.actor(obs_x, actions_fixed, actions_discard)
        
        #Categorical 是 PyTorch 中用于建模离散分类分布的类。
        # Step 1: 选择固定牌型
        distribution_fixed = Categorical(actions_fixed_prob)
        action_fixed = distribution_fixed.sample()
        log_prob_fixed = distribution_fixed.log_prob(action_fixed).sum(-1)# 如果动作是多维的（n_actions > 1），则 log_prob 返回的是每个维度的对数概率，需要用 .sum(-1) 合并成一个标量。
        action_fixed = action_fixed.detach()

        # Step 2: 选择垫牌        
        distribution_discard = Categorical(discard_action_prob)
        actions_discard = t.multinomial(discard_action_prob, num_samples=discard_num, replacement=False)#无放回采样
        log_probs_discard = distribution_discard.log_prob(actions_discard).sum(-1)# [batch_size, k]
        actions_discard = actions_discard.detach()
        
        # Step 3: 返回固定牌型动作，弃牌动作，log prob总和
        total_logprob = log_prob_fixed + log_probs_discard  # sum over multiple discards

        return action_fixed, actions_discard[action_fixed], pre_publiccard, log_prob_fixed.detach(), log_probs_discard.detach(), total_logprob.detach()
    

    
        #sample() 方法会从这个分布中按概率随机采样，鼓励探索, sample只支持有放回采样，所以不适用        
        # action = distribution.sample()
        # action_logprob = distribution.log_prob(action)
        
        #对于连续动作空间可以使用高斯分布（torch.distributions.Normal ）来建模策略
        # mean = self.forward(obs_x)
        # std = t.exp(self.log_std)  # 转换为标准差 σ
        # distribution = t.distributions.Normal(mean, std)
        # action = distribution.sample()
        # action_logprob = distribution.log_prob(action).sum(-1)  # 如果动作是多维的（n_actions > 1），则 log_prob 返回的是每个维度的对数概率，需要用 .sum(-1) 合并成一个标量。
        # return action.detach(), action_logprob.detach()
    
    def evaluate(self, obs_x, actions_fixed, actions_discard, discard_num):
        actions_fixed_prob, discard_action_prob, pre_publiccard, log_probs_publiccard, entropy_publiccard = self.forward(obs_x, actions_fixed, actions_discard)

        #固定组合牌
        distribution_fixed = Categorical(actions_fixed_prob)
        action_fixed = distribution_fixed.sample()
        log_prob_fixed = distribution_fixed.log_prob(action_fixed).sum(-1)# 如果动作是多维的（n_actions > 1），则 log_prob 返回的是每个维度的对数概率，需要用 .sum(-1) 合并成一个标量。
        entropy_fixed = distribution_fixed.entropy().sum(-1)#连续动作空间的熵
        # action_fixed = action_fixed.detach()

        #垫牌
        distribution_discard = Categorical(discard_action_prob)
        actions_discard = t.multinomial(discard_action_prob, num_samples=discard_num, replacement=False)#无放回采样
        log_probs_discard = distribution_discard.log_prob(actions_discard).sum(-1)# [batch_size, k]
        entropy_discard = distribution_discard.entropy().sum(-1)
        # actions_discard = actions_discard.detach()
        
        #状态评价
        state_value = self.critic.forward(obs_x)
        return log_prob_fixed, log_probs_discard, state_value, entropy_fixed, entropy_discard
        # mean = self.actor.forward(state)
        # std = t.exp(self.actor.log_std)
        # distribution = t.distributions.Normal(mean, std)
        # action_log_prob = distribution.log_prob(action).sum(-1)
        # entropy = distribution.entropy().sum(-1)#连续动作空间的熵
        # state_value = self.critic.forward(state)
        # return action_log_prob, state_value, entropy
    
    def act_all_probs(self, x: Tensor):
        action_prob = self.forward(x)
        distribution = Categorical(action_prob)
        action = distribution.sample()
        return action.detach().item(), action_prob.cpu().detach().numpy()



    def generate_batches(self, discounted_rewards: np.ndarray, batch_size: int) -> np.ndarray:
        """
        generate many batches of samples from the buffer,
        note that the last batch's length may be smaller than batch_size
        """
        one_left = len(discounted_rewards) % batch_size == 1
        n_states = len(discounted_rewards)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)

        batches = [indices[i:i+batch_size]
                   for i in batch_start if len(indices[i:i+batch_size]) > 1]
        if one_left:
            batches[-1] = np.concatenate((batches[-1], indices[-1:]))
        return batches

    def update(self):
        """
        update the weights of the actor and critic using PPOClip
        -------
        Parameters:
            epochs: int, how many gradient descent epochs do you want to perform?
            batch_mode: bool, set True if you want to activate batch mode, False otherwise
            batch_size: int
        Returns:
            eploss_a: float, the mean loss of the actor in the update procedure
            eploss_c: float, the mean loss of the critic in the update procedure
        """
        # generate batch data
        data = self.get_data()
        discounted_rewards = data["discounted_rewards"]
        batch_size = self.batch_size if self.batch_mode else len(
            discounted_rewards)
        batches = self.generate_batches(discounted_rewards, batch_size)

        # start training procedure
        ep_lossa, ep_lossc = [], []
        for ep in range(self.epochs):
            for batch in batches:
                # use from numpy instead of torch.tensor() to accelerate
                discounted_rewards = t.from_numpy(
                    data["discounted_rewards"][batch]).to(self.device)
                # normalize the advantages
                advantages = t.from_numpy(
                    data["advantages"][batch]).to(self.device)
                advantages = (advantages-advantages.mean()) / \
                    (advantages.std()+1e-10)

                # convert ndarray to tensor
                old_states = t.from_numpy(
                    data["states"][batch]).to(self.device)
                old_actions = t.from_numpy(
                    data["actions"][batch]).to(self.device)
                old_log_probs = t.from_numpy(
                    data["log_probs"][batch]).to(self.device)

                # get the value from the new networks, we only use log_probs and state_values
                # as part of the computation graph to update the weights of actor and critic
                log_probs, state_values, dist_entropy = self.evaluate(
                    old_states, old_actions)
                state_values = state_values.squeeze()

                # calculate the surrogate loss
                ratios = t.exp(log_probs-old_log_probs.detach())
                surr1 = ratios*advantages
                surr2 = t.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip)*advantages
                # calculate the actor loss
                loss_actor = t.mean(-t.min(surr1, surr2) -
                                    self.entropy_weight*dist_entropy)
                # calculate the cirtic loss
                loss_critic = self.mseloss.forward(
                    state_values, discounted_rewards)
                loss = loss_actor + loss_critic
                # zero_grad and step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # record the loss
                ep_lossa.append(loss_actor.item())
                ep_lossc.append(loss_critic.item())
        # copy parameters
        self.update_old_net()
        # clean the buffer
        self.clean_buffer()
        return sum(ep_lossa)/len(ep_lossa), sum(ep_lossc)/len(ep_lossc)

    def learn(self, env: gym.Env):
        """
        start to train the actor critic
        """
        time_steps = 0
        current_episode_reward = 0
        while time_steps < self.training_epochs:
            print(
                f"{time_steps}/{self.training_epochs},reward:{current_episode_reward}")

            state = env.reset()
            if isinstance(state, tuple): state = state[0]
            current_episode_reward = 0

            for ts in range(1, self.max_episode_length+1):
                # without gradient
                with t.no_grad():
                    state = t.from_numpy(state).to(self.device)
                    # get action and log probability
                    action, log_prob = self.old_actor.act(state)
                    # get critic evaluation of the state
                    value = self.old_critic.forward(state)
                # take a step in the environment
                next_state, reward, done, _, _ = env.step(action.item())

                # record loss and add timestep
                current_episode_reward += reward
                time_steps += 1

                timeout = (ts == self.max_episode_length)
                update = time_steps % self.update_interval == 0
                terminal = done or timeout or update
                # store the transition
                self.store_transition(
                    reward, done, state, action, log_prob, value)

                if terminal:
                    # if not done, calculate the value of next state
                    if not done:
                        with t.no_grad():
                            value = self.old_critic(
                                t.from_numpy(next_state).to(self.device)).cpu()
                    else:
                        value = 0
                    # calculate the advantages and returns
                    self.finish_path(value)
                    # if update interval is reached, update the parameters
                    if update:
                        loss_a, loss_c = self.update()
                        self.save_parameters(time_steps)
                        with open(os.path.join(self.log_path, "log.txt"), "a") as f:
                            f.write(
                                f"{time_steps},{loss_a},{loss_c}\n")
                    if done:
                        break
                state = next_state

    def update_old_net(self):
        """
        copy the parameters of the new network to the old network
        """
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())

    def save_parameters(self, ep):
        t.save(self.old_actor.state_dict(),
               os.path.join(self.save_path, f"actor_{ep}.pth"))
        t.save(self.old_critic.state_dict(),
               os.path.join(self.save_path, f"critic_{ep}.pth"))

    def load_parameters(self):
        self.actor.load_state_dict(t.load(self.actor_path))
        self.critic.load_state_dict(t.load(self.critic_path))
        self.update_old_net()

    def test(self, env: gym.Env):
        for ep in range(1,30):
            state,_ = env.reset()
            current_episode_reward = 0
            env.render()
            
            for ts in range(1, self.max_episode_length+1):
                with t.no_grad():
                    state = t.from_numpy(state).to(self.device)
                    action, log_prob = self.old_actor.act(state)
                next_state, reward, done, _,_ = env.step(action.item())
                state = next_state
                done = True if ts == self.max_episode_length else done
                env.render()
                current_episode_reward += reward
                if done:
                    break
            print(f"ep:{ep} reward:{current_episode_reward}")




class PlayModel(nn.Module):
    def __init__(self):
        super().__init__()

    def load_checkpoint(self):
        pass