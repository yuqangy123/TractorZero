import matplotlib.pyplot as plt
import numpy as np

# 全局配置（论文级格式）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['savefig.bbox_inches'] = 'tight'
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.8

# 数据准备（贴合论文描述）
stages = ['前期（1-5轮）', '中期（6-10轮）', '后期（11轮后）']
x = np.arange(len(stages))
width = 0.35

# 回合胜率（%）
win_rate_nc = [50.2, 60.5, 65.3]  # 有信息熵降噪（Nc Model）
win_rate_nonc = [46.1, 50.3, 64.8]  # 无降噪（NoNc Model）
# 标准差
std_win_nc = [2.1, 1.8, 1.5]
std_win_nonc = [2.5, 2.2, 1.6]

# 绘制分组柱状图
def plot_round_win_rate():
    fig, ax = plt.subplots(figsize=(8, 5))

    # 配色（与前文一致）
    color_nc = '#C73E1D'
    color_nonc = '#F18F01'

    # 绘制柱状图+误差棒
    bars1 = ax.bar(x - width/2, win_rate_nc, width, label='Nc Model（有信息熵降噪）',
                   color=color_nc, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.errorbar(x - width/2, win_rate_nc, yerr=np.array(std_win_nc)*1.96,
                fmt='none', color='black', capsize=5, capthick=1.5)

    bars2 = ax.bar(x + width/2, win_rate_nonc, width, label='NoNc Model（无降噪）',
                   color=color_nonc, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.errorbar(x + width/2, win_rate_nonc, yerr=np.array(std_win_nonc)*1.96,
                fmt='none', color='black', capsize=5, capthick=1.5)

    # 标注数值
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_labels(bars1)
    add_labels(bars2)

    # 轴标签与标题
    ax.set_xlabel('牌局阶段', fontsize=11)
    ax.set_ylabel('回合胜率（%）', fontsize=11)
    ax.set_title('信息熵降噪对不同牌局阶段回合胜率的影响（对战RuleRobot+RuleRobot）',
                 fontsize=12, fontweight='bold', pad=20)

    # 坐标轴设置
    ax.set_ylim(40, 70)
    ax.set_yticks(np.arange(40, 71, 5))
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.grid(True, axis='y', linestyle='--')

    # 图例
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)

    # 保存
    plt.savefig('图5.4.3.3-1 不同牌局阶段回合胜率对比图.pdf', format='pdf', dpi=300)
    plt.close()

# 执行绘图
plot_round_win_rate()
print("✅ 图5.4.3.3-1已生成完成")