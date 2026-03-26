import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches

# ---------------------- 1. 数据准备（完全基于论文统计结果）----------------------
# 核心数据：(经度, 纬度, 人才数量权重)
talent_data = [
    # 美国核心区域
    (-122.0, 37.4, 12),   # 硅谷
    (-71.0, 42.3, 5),     # 波士顿
    (-74.0, 40.7, 3),     # 纽约
    (-95.0, 37.0, 8),     # 美国其他区域
    
    # 中国核心区域
    (114.0, 30.5, 14),    # 武汉
    (116.4, 39.9, 3),     # 北京
    (114.0, 22.5, 2),     # 深圳
    (121.5, 31.2, 1),     # 上海
    
    # 其他国家/区域
    (8.0, 47.0, 5),       # 欧洲
    (104.0, 1.3, 2),      # 新加坡
    (134.0, -25.0, 1)     # 澳大利亚
]

# 拆分经度、纬度、权重数组
lon = np.array([x[0] for x in talent_data])
lat = np.array([x[1] for x in talent_data])
weights = np.array([x[2] for x in talent_data])

# ---------------------- 2. 生成网格数据（优化计算逻辑，解决NumPy警告）----------------------
# 创建全球范围的经纬度网格
lon_range = np.linspace(-180, 180, 720)
lat_range = np.linspace(-60, 80, 560)
lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

# 初始化核密度估计器（只初始化一次，提升效率）
kde = gaussian_kde(np.vstack([lon, lat]), weights=weights)

# 优化：将网格展平后批量计算，避免双重循环，同时提取标量
# 展平网格为二维数组 (2, N)
grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
# 批量计算密度值
z_vals = kde(grid_points)
# 重塑为网格形状（核心修复：z_vals是一维数组，直接重塑即可）
z_grid = z_vals.reshape(lon_grid.shape)

# ---------------------- 3. 绘制热力图 ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False    # 支持负号

fig, ax = plt.subplots(figsize=(16, 8))

# 优先尝试使用cartopy绘制真实地理轮廓
use_cartopy = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax = plt.axes(projection=ccrs.PlateCarree())
    # 添加海岸线和陆地轮廓
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black', alpha=0.6)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5)
    use_cartopy = True
except ImportError:
    # 未安装cartopy时使用基础模式
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.grid(True, alpha=0.3, linestyle='--')

# 绘制热力图（适配cartopy投影）
heatmap = ax.imshow(
    z_grid, 
    extent=(-180, 180, -60, 80),
    origin='lower',
    cmap='jet',
    alpha=0.7,
    aspect='auto',
    transform=ccrs.PlateCarree() if use_cartopy else None
)

# ---------------------- 4. 添加标注和图例 ----------------------
# 标注核心城市（人才数量≥2人）
core_cities = [
    (-122.0, 37.4, '硅谷 (12人)'),
    (-71.0, 42.3, '波士顿 (5人)'),
    (114.0, 30.5, '武汉 (14人)'),
    (116.4, 39.9, '北京 (3人)'),
    (114.0, 22.5, '深圳 (2人)')
]
for lon_c, lat_c, label in core_cities:
    ax.annotate(
        label, 
        xy=(lon_c, lat_c),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        transform=ccrs.PlateCarree() if use_cartopy else None
    )

# 添加颜色条
cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8)
cbar.set_label('人才集聚强度', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# 设置标题和坐标轴标签
ax.set_title(
    '“卡脖子”领域湖北籍海外高校理工科人才全球分布热力图\n（n=56，数据来源：论文履历深度挖掘）',
    fontsize=14, fontweight='bold', pad=20
)
if not use_cartopy:
    ax.set_xlabel('经度', fontsize=12, fontweight='bold')
    ax.set_ylabel('纬度', fontsize=12, fontweight='bold')

# 添加图例
legend_elements = [
    mpatches.Patch(color='#FF0000', alpha=0.7, label='核心集聚地（硅谷、武汉）'),
    mpatches.Patch(color='#FF9900', alpha=0.7, label='重要集聚地（波士顿、北京）'),
    mpatches.Patch(color='#0066FF', alpha=0.7, label='一般集聚地（欧洲、新加坡）')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# ---------------------- 5. 保存图片 ----------------------
plt.tight_layout()
plt.savefig('人才分布热力图_修复版2.png', dpi=300, bbox_inches='tight')
plt.show()

print("热力图已保存为：人才分布热力图_修复版2.png")