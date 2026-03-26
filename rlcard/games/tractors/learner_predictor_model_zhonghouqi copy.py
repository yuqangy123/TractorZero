import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches

# ---------------------- 1. 数据准备（完全基于论文统计结果）----------------------
# 核心数据：(经度, 纬度, 人才数量权重)
# 权重依据论文中各区域/城市的人才数量比例设定
talent_data = [
    # 美国核心区域（硅谷、波士顿、纽约）
    (-122.0, 37.4, 12),   # 硅谷（12人，最高权重）
    (-71.0, 42.3, 5),     # 波士顿（5人）
    (-74.0, 40.7, 3),     # 纽约（3人）
    (-95.0, 37.0, 8),     # 美国其他区域（28-12-5-3=8人）
    
    # 中国核心区域（湖北武汉及国内其他城市）
    (114.0, 30.5, 14),    # 武汉（14人，次高权重）
    (116.4, 39.9, 3),    # 北京（3人）
    (114.0, 22.5, 2),     # 深圳（2人）
    (121.5, 31.2, 1),     # 上海（1人）
    
    # 其他国家/区域
    (8.0, 47.0, 5),       # 欧洲（5人）
    (104.0, 1.3, 2),      # 新加坡（2人）
    (134.0, -25.0, 1)     # 澳大利亚（1人）
]

# 拆分经度、纬度、权重数组
lon = np.array([x[0] for x in talent_data])
lat = np.array([x[1] for x in talent_data])
weights = np.array([x[2] for x in talent_data])

# ---------------------- 2. 生成网格数据（用于热力图平滑渲染）----------------------
# 创建全球范围的经纬度网格（步长0.5°，保证分辨率）
lon_range = np.linspace(-180, 180, 720)
lat_range = np.linspace(-60, 80, 560)
lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

# 使用高斯核密度估计生成热力图数据（模拟人才集聚效果）
xy = np.vstack([lon, lat])
z = gaussian_kde(xy, weights=weights)(xy)
# 对网格进行密度计算
z_grid = np.zeros(lon_grid.shape)
for i in range(len(lon_range)):
    for j in range(len(lat_range)):
        z_grid[j, i] = gaussian_kde(xy, weights=weights)([lon_range[i], lat_range[j]])

# ---------------------- 3. 绘制热力图 ----------------------
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
# plt.rcParams['axes.unicode_minus'] = False    # 支持负号

fig, ax = plt.subplots(figsize=(16, 8))

# 绘制热力图（使用jet色彩映射，突出双核心）
heatmap = ax.imshow(
    z_grid, 
    extent=(-180, 180, -60, 80),
    origin='lower',
    cmap='jet',
    alpha=0.8,
    aspect='auto'
)

# # 添加海岸线（增强地理辨识度）
# try:
#     # 若安装了cartopy库，可添加真实海岸线（可选）
#     import cartopy.crs as ccrs
#     import cartopy.feature as cfeature
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black', alpha=0.6)
#     ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
#     # 重新绘制热力图（适配cartopy投影）
#     heatmap = ax.imshow(z_grid, extent=(-180, 180, -60, 80), origin='lower', cmap='jet', alpha=0.7)
# except ImportError:
#     # 未安装cartopy时，使用基础地理轮廓
#     ax.set_xlim(-180, 180)
#     ax.set_ylim(-60, 80)
#     ax.grid(True, alpha=0.3, linestyle='--')

# ---------------------- 4. 添加标注和图例 ----------------------
# 标注核心城市（人才数量≥2人）
core_cities = [
    (-122.0, 37.4, 'Silicon Valley (12 people)'),
    (-71.0, 42.3, 'Boston (5 people)'),
    (114.0, 30.5, 'Wuhan (14 people)'),
    (116.4, 39.9, 'Beijing (3 people)'),
    (114.0, 22.5, 'Shenzhen (2 people)')
]
for lon_c, lat_c, label in core_cities:
    ax.annotate(
        label, 
        xy=(lon_c, lat_c),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

# 添加颜色条（标注人才集聚强度）
cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8)
cbar.set_label('人才集聚强度', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# 设置标题和坐标轴标签
ax.set_title(
    '“',
    fontsize=14, fontweight='bold', pad=20
)
ax.set_xlabel('经度', fontsize=12, fontweight='bold')
ax.set_ylabel('纬度', fontsize=12, fontweight='bold')

# 添加图例（说明双核心特征）
legend_elements = [
    mpatches.Patch(color='#FF0000', alpha=0.7, label='Core gathering places (Silicon Valley, Wuhan)'),
    mpatches.Patch(color='#FF9900', alpha=0.7, label='Important gathering places (Boston, Beijing)'),
    mpatches.Patch(color='#0066FF', alpha=0.7, label='General gathering places (Europe, Singapore)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# ---------------------- 5. 保存图片 ----------------------
plt.tight_layout()
plt.savefig('人才分布热力图.png', dpi=300, bbox_inches='tight')
plt.show()

print("热力图已保存为：人才分布热力图.png")