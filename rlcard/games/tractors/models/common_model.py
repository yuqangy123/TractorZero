import torch as t
import torch.nn as nn
from rlcard.games.tractors.models.stractor_resnet import ResNet, ResidualBlock

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
        
        # 手牌特征提取器 (2,4,14) -> 256维
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
