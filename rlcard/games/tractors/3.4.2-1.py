import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（适配论文排版）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据（剩余手牌数16-1，预测差距均值±标准差）
remaining_cards = np.arange(16, 0, -1)
mean_gap = [3.8, 3.6, 3.4, 3.2, 2.9, 2.7, 2.4, 2.2, 1.9, 1.7, 1.4, 1.2, 0.69, 0.38, 0.15, 0.1]
std_gap = [0.42, 0.38, 0.35, 0.33, 0.30, 0.28, 0.25, 0.23, 0.21, 0.19, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07]

# 创建图表（论文标准尺寸6.4×4.8英寸）
fig, ax = plt.subplots(figsize=(6.4, 4.8))

# 绘制带误差线的折线图
ax.errorbar(remaining_cards, mean_gap, yerr=std_gap, fmt='-o', color='#1f77b4', 
            linewidth=2, markersize=6, capsize=5, capthick=1, ecolor='#2c3e50')

# 设置坐标轴范围与标签
ax.set_xlim(0.5, 16.5)
ax.set_ylim(0, 4.5)
ax.set_xlabel('剩余手牌数（张）', fontsize=10, fontweight='bold')
ax.set_ylabel('花色数量误差均值', fontsize=10, fontweight='bold')
ax.set_title('不同剩余手牌数时对家手牌花色数量预测误差均值', fontsize=12, fontweight='bold', pad=20)

# 设置网格线（学术图表常用样式）
ax.grid(True, axis='y', linestyle='--', alpha=0.6, color='#95a5a6')
ax.set_axisbelow(True)

# 调整横轴刻度标签
ax.set_xticks(remaining_cards[::])  # 每2个刻度显示一次，避免拥挤
ax.set_xticklabels(remaining_cards[::], fontsize=9)
ax.set_yticks(np.arange(0, 4.1, 0.5))
ax.set_yticklabels(np.arange(0, 4.1, 0.5), fontsize=9)

# 添加图注（放在图表下方，适配论文格式）
plt.figtext(0.5, 0.01, '注：预测差距为模型输出的对家4种花色手牌数量与真实值的绝对误差均值，误差线为标准差；样本量为16,000个对局样本。', 
            ha='center', fontsize=8, wrap=True)

# 紧凑布局，保存为高清矢量图（论文常用EPS格式）
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('隐藏信息预测差距折线图.eps', dpi=300, bbox_inches='tight')
plt.show()