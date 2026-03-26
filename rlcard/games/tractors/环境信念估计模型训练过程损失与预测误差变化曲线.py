import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 全局配置：设置中文字体、图表样式
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文和英文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.style.use('seaborn-v0_8-notebook')  # 美观的图表风格

# --------------------------
# 图1：环境信念估计模型训练曲线（对应4.2.3.1）
# 包含：训练/验证损失、花色分布预测距离、底牌分数预测距离
# --------------------------
def plot_belief_train_curve():
    epochs = np.arange(0, 501, 50)  # 0-500 epoch，每50步取一个点
    # 模拟训练/验证损失（先快速下降后平稳）
    train_loss = [0.92, 0.68, 0.45, 0.32, 0.23, 0.18, 0.16, 0.15, 0.148, 0.145, 0.143]
    val_loss = [0.95, 0.72, 0.49, 0.36, 0.27, 0.22, 0.19, 0.17, 0.165, 0.162, 0.16]
    # 模拟预测距离（归一化后，越小越优）
    suit_dist = [0.85, 0.62, 0.41, 0.30, 0.25, 0.23, 0.21, 0.205, 0.202, 0.20, 0.198]
    bottom_dist = [0.90, 0.68, 0.46, 0.35, 0.29, 0.26, 0.24, 0.23, 0.225, 0.22, 0.218]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制损失曲线（左y轴）
    ax1.plot(epochs, train_loss, 'o-', color='#2E86AB', linewidth=2.5, markersize=6, label='训练损失')
    ax1.plot(epochs, val_loss, 's-', color='#A23B72', linewidth=2.5, markersize=6, label='验证损失')
    ax1.set_xlabel('训练轮次（Epoch）', fontsize=12, fontweight='bold')
    ax1.set_ylabel('损失值（Loss）', fontsize=12, fontweight='bold', color='#2E86AB')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)

    # 绘制预测距离（右y轴）
    ax2 = ax1.twinx()
    ax2.plot(epochs, suit_dist, '^-', color='#F18F01', linewidth=2.5, markersize=6, label='花色分布预测距离')
    ax2.plot(epochs, bottom_dist, 'd-', color='#C73E1D', linewidth=2.5, markersize=6, label='底牌分数预测距离')
    ax2.set_ylabel('预测距离（归一化）', fontsize=12, fontweight='bold', color='#F18F01')
    ax2.tick_params(axis='y', labelcolor='#F18F01')
    ax2.set_ylim(0, 1.0)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.title('环境信念估计模型训练曲线', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('belief_train_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------
# 图2：分层强化学习收敛对比（对应5.4.3.1）
# 包含：HRL+Belief vs HRL+w/oBelief 的胜率和场均得分
# --------------------------
def plot_hrl_convergence():
    batches = np.arange(0, 120001, 10000)  # 0-12万批次
    # 模拟胜率（%）
    hrl_belief_win = [35, 48, 62, 75, 83, 87, 88.5, 89.2, 89.5, 89.7, 89.7, 89.7]
    hrl_no_belief_win = [32, 42, 55, 65, 72, 76, 79, 81, 82, 82.3, 82.5, 82.5]
    # 模拟场均得分
    hrl_belief_score = [55, 68, 82, 90, 95, 97, 98, 98.3, 98.5, 98.5, 98.5, 98.5]
    hrl_no_belief_score = [50, 60, 70, 78, 82, 84, 85, 85.5, 86, 86.2, 86.2, 86.2]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 子图1：胜率对比
    ax1.plot(batches, hrl_belief_win, 'o-', color='#2E86AB', linewidth=3, markersize=6, label='HRL+Belief')
    ax1.plot(batches, hrl_no_belief_win, 's-', color='#C73E1D', linewidth=3, markersize=6, label='HRL+w/oBelief')
    ax1.set_ylabel('胜率（%）', fontsize=12, fontweight='bold')
    ax1.set_ylim(30, 95)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_title('训练收敛对比 - 胜率', fontsize=13, fontweight='bold')

    # 子图2：场均得分对比
    ax2.plot(batches, hrl_belief_score, 'o-', color='#2E86AB', linewidth=3, markersize=6, label='HRL+Belief')
    ax2.plot(batches, hrl_no_belief_score, 's-', color='#C73E1D', linewidth=3, markersize=6, label='HRL+w/oBelief')
    ax2.set_xlabel('训练批次（Batch）', fontsize=12, fontweight='bold')
    ax2.set_ylabel('场均得分', fontsize=12, fontweight='bold')
    ax2.set_ylim(45, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_title('训练收敛对比 - 场均得分', fontsize=13, fontweight='bold')

    plt.suptitle('联合环境信念估计对收敛速度的提升', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('hrl_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------
# 图3：信息熵降噪效果对比（对应5.4.3.3）
# 包含：不同牌局阶段的回合胜率 + 最终得分
# --------------------------
def plot_entropy_denoise():
    stages = ['前期（1-5轮）', '中期（6-10轮）', '后期（11轮后）']
    nc_win_rate = [50.3, 60.5, 64.8]  # 有降噪
    no_nc_win_rate = [46.1, 50.2, 65.3]  # 无降噪
    final_score = [97.1, 89.5]  # 最终得分对比
    score_labels = ['有信息熵降噪', '无降噪处理']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 子图1：不同阶段回合胜率
    x = np.arange(len(stages))
    width = 0.35
    ax1.bar(x - width/2, nc_win_rate, width, label='有信息熵降噪', color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, no_nc_win_rate, width, label='无降噪处理', color='#C73E1D', alpha=0.8)
    ax1.set_xlabel('牌局阶段', fontsize=12, fontweight='bold')
    ax1.set_ylabel('回合胜率（%）', fontsize=12, fontweight='bold')
    ax1.set_ylim(40, 70)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=15)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_title('不同牌局阶段的回合胜率对比', fontsize=13, fontweight='bold')

    # 子图2：最终得分对比
    ax2.bar(score_labels, final_score, color=['#2E86AB', '#C73E1D'], alpha=0.8, width=0.5)
    ax2.set_ylabel('牌局最终得分', fontsize=12, fontweight='bold')
    ax2.set_ylim(85, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title('牌局最终得分对比', fontsize=13, fontweight='bold')
    # 在柱状图上添加数值标签
    for i, v in enumerate(final_score):
        ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('信息熵置信度降噪效果分析', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('entropy_denoise_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------
# 执行绘图函数（按需调用）
# --------------------------
if __name__ == "__main__":
    plot_belief_train_curve()  # 环境信念估计模型训练曲线
    plot_hrl_convergence()     # 分层强化学习收敛对比
    plot_entropy_denoise()     # 信息熵降噪效果对比