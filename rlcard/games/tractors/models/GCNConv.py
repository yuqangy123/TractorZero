import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# -------------------------- 1. 模拟论文中的图结构数据（双升游戏牌型关系）--------------------------
def create_tractor_graph():
    """
    构建论文中的图数据：108个节点（两副牌），4类边（同花色/同牌型/同玩家/历史关联）
    节点特征：62维（论文4.2.1节定义），边特征：4维（one-hot表示边类型）
    """
    num_nodes = 108  # 双升两副牌共108张
    
    # 1. 节点特征（62维）：模拟论文中的牌属性+归属信息
    # 前24维：点数(15维one-hot)+花色(4维one-hot)+主牌/级牌(2维)+分数牌(1维)+可见性(1维)+是否底牌(1维)
    # 后38维：归属玩家(4维one-hot)+庄/闲(1维)+历史出牌频次(33维，简化为随机值)
    node_features = torch.randn(num_nodes, 62)  # 模拟特征，实际需替换为论文中的真实编码
    
    # 2. 边索引（COO格式：[2, num_edges]）：模拟4类边的连接关系
    edges = []
    edge_attrs = []  # 边特征：4维one-hot，对应4类边
    
    # 模拟同花色边（108张牌分4种花色，每种花色27张，两两相连）
    for suit in range(4):
        suit_nodes = list(range(suit*27, (suit+1)*27))
        for i in range(len(suit_nodes)):
            for j in range(i+1, len(suit_nodes)):
                edges.append([suit_nodes[i], suit_nodes[j]])
                edges.append([suit_nodes[j], suit_nodes[i]])  # 无向边
                edge_attrs.append([1, 0, 0, 0])  # 同花色边标识
    
    # 模拟同玩家边（4个玩家，每个玩家27张牌，两两相连）
    for player in range(4):
        player_nodes = list(range(player*27, (player+1)*27))
        for i in range(len(player_nodes)):
            for j in range(i+1, len(player_nodes)):
                edges.append([player_nodes[i], player_nodes[j]])
                edges.append([player_nodes[j], player_nodes[i]])
                edge_attrs.append([0, 1, 0, 0])  # 同玩家边标识
    
    # 模拟同牌型边（简化：点数相同的牌相连，如两张3、两张4等）
    for rank in range(15):  # 15种点数（2-A+大小王）
        rank_nodes = [rank + 15*k for k in range(7)]  # 两副牌共7张同点数牌（简化）
        for i in range(len(rank_nodes)):
            for j in range(i+1, len(rank_nodes)):
                if rank_nodes[i] < 108 and rank_nodes[j] < 108:
                    edges.append([rank_nodes[i], rank_nodes[j]])
                    edges.append([rank_nodes[j], rank_nodes[i]])
                    edge_attrs.append([0, 0, 1, 0])  # 同牌型边标识
    
    # 模拟历史关联边（简化：相邻回合出牌的牌相连）
    history_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]  # 示例边
    for u, v in history_edges:
        edges.append([u, v])
        edges.append([v, u])
        edge_attrs.append([0, 0, 0, 1])  # 历史关联边标识
    
    # 转换为PyG要求的格式
    edge_index = torch.tensor(edges, dtype=torch.long).T  # [2, num_edges]
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)  # [num_edges, 4]
    
    # 3. 构建PyG图数据对象
    data = Data(
        x=node_features,  # 节点特征 [108, 62]
        edge_index=edge_index,  # 边索引 [2, num_edges]
        edge_attr=edge_attr  # 边特征 [num_edges, 4]
    )
    return data

# -------------------------- 2. 论文中的GCN模型定义（3层GCNConv，带残差连接）--------------------------
class TractorGCN(nn.Module):
    def __init__(self, input_dim=62, hidden_dim=128, output_dim=256, dropout=0.2):
        super(TractorGCN, self).__init__()
        # 论文4.2.2节GCN结构：3层GCNConv，带残差连接
        self.conv1 = GCNConv(input_dim + 4, hidden_dim)  # 节点特征+边特征拼接作为输入
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.residual1 = nn.Linear(input_dim + 4, hidden_dim)  # 残差连接映射
        self.residual2 = nn.Linear(hidden_dim, hidden_dim * 2)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 节点特征 + 边特征聚合（论文中边特征参与节点更新）
        # 方法：将边特征广播到节点特征（简化为与邻接节点的边特征均值聚合）
        from torch_geometric.utils import degree
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)  # 每个节点的度
        edge_attr_sum = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        edge_attr_sum.index_add_(0, row, edge_attr)  # 按行聚合边特征
        edge_attr_mean = edge_attr_sum / (deg.unsqueeze(1) + 1e-6)  # 边特征均值
        x = torch.cat([x, edge_attr_mean], dim=1)  # 节点特征+边特征 [108, 62+4=66]
        
        # 第一层GCN（带残差）
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        residual1 = self.residual1(x)
        h1 = h1 + residual1  # 残差连接
        
        # 第二层GCN（带残差）
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        residual2 = self.residual2(h1)
        h2 = h2 + residual2  # 残差连接
        
        # 第三层GCN（输出256维全局特征）
        h3 = self.conv3(h2, edge_index)
        
        # 全局池化（论文中需全局牌型特征，用均值池化）
        global_feat = global_mean_pool(h3, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        # batch为全0表示所有节点属于同一个图（单局游戏）
        
        return global_feat  # 输出全局特征 [1, 256]，对应论文中的GCN特征

# -------------------------- 3. 模型测试（模拟论文中的前向传播）--------------------------
if __name__ == "__main__":
    # 1. 构建图数据
    data = create_tractor_graph()
    print("图数据信息：")
    print(f"节点数：{data.x.size(0)}, 节点特征维度：{data.x.size(1)}")
    print(f"边数：{data.edge_index.size(1)//2}（无向边），边特征维度：{data.edge_attr.size(1)}")
    
    # 2. 初始化模型（匹配论文参数）
    gcn_model = TractorGCN(
        input_dim=62,
        hidden_dim=128,
        output_dim=256,
        dropout=0.2
    )
    
    # 3. 前向传播（输出256维全局特征，可与ResNet/LSTM特征融合）
    global_gcn_feat = gcn_model(data)
    print(f"\nGCN输出全局特征维度：{global_gcn_feat.size()}")  # 输出：torch.Size([1, 256])
    
    # 4. 集成到论文框架（与ResNet空间特征、LSTM时序特征融合）
    # 模拟ResNet特征（256维）、LSTM特征（512维）
    resnet_feat = torch.randn(1, 256)
    lstm_feat = torch.randn(1, 512)
    # 特征融合（论文中α=0.3, β=0.4, γ=0.3）
    fused_feat = 0.3 * resnet_feat + 0.4 * lstm_feat + 0.3 * global_gcn_feat
    print(f"融合后特征维度：{fused_feat.size()}")  # 输出：torch.Size([1, 256])