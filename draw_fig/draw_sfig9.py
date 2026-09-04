import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

# =========================================================
# 1. data
# =========================================================

df_fold = pd.read_csv('../results/ridge/fold/auc.csv',header=None)
df_trans = pd.read_csv('../results/ridge/trans/auc.csv',header=None)
df_sup_hopf = pd.read_csv('../results/ridge/sup_hopf/auc.csv',header=None)
df_sub_hopf = pd.read_csv('../results/ridge/sub_hopf/auc.csv',header=None)
df_fold_ER = pd.read_csv('../results/ridge/fold_ER/auc.csv',header=None)
df_fold_BA = pd.read_csv('../results/ridge/fold_BA/auc.csv',header=None)

fold_auc = df_fold.iloc[1:, 1:].to_numpy(dtype=float)
trans_auc = df_trans.iloc[1:, 1:].to_numpy(dtype=float)
sup_hopf_auc = df_sup_hopf.iloc[1:, 1:].to_numpy(dtype=float)
sub_hopf_auc = df_sub_hopf.iloc[1:, 1:].to_numpy(dtype=float)
fold_ER_auc = df_fold_ER.iloc[1:, 1:].to_numpy(dtype=float)
fold_BA_auc = df_fold_BA.iloc[1:, 1:].to_numpy(dtype=float)

data_list = [fold_auc, trans_auc, sup_hopf_auc, sub_hopf_auc, fold_ER_auc, fold_BA_auc]

titles = ['Intraspecific competition','Predator-prey (transcritical)','Predator-prey (Hopf)',
    'Epidemic','Mutualistic network (ER)','Mutualistic network (BA)']

# x, y 每个格子的标签
x_labels = ['0.001', '0.005', '0.01', '0.05', '0.1']
y_labels = ['0.05', '0.04', '0.03', '0.02', '0.01']

# =========================================================

plt.rcParams.update({
    "font.family": "Arial",

    "font.size": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,

    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

# =========================================================
# 3. 建图
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
fig.patch.set_facecolor("white")

# 统一颜色范围
# vmin = min(np.min(d) for d in data_list)
# vmax = max(np.max(d) for d in data_list)
vmin = 0.5
vmax = 1

cmap = "cividis"
# cmap = "Blues"

for i, ax in enumerate(axes.flat):
    data = data_list[i]

    im = ax.imshow(
        data,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal"
    )

    # 标题
    ax.set_title(titles[i], fontsize=15, fontweight='normal', fontfamily='DejaVu Sans',fontstyle= 'italic', y=1.02)

    # =========================
    # 高亮：最下面一行 / 中间一列 / 交汇格子
    # =========================
    nrows, ncols = data.shape
    bottom_row = nrows - 1
    middle_col = ncols // 2   # 5列时就是第3列，索引为2

    # 1) 最下面一行：虚线框
    rect_bottom = Rectangle(
        (-0.5, bottom_row - 0.5),   # 左下角
        ncols,                      # 宽度
        1,                          # 高度
        fill=False,
        edgecolor="black",
        linewidth=2,
        linestyle="--",
        zorder=2,
        clip_on=False
    )
    ax.add_patch(rect_bottom)

    # # 2) 中间那一列：点线框
    # rect_middle = Rectangle(
    #     (middle_col - 0.5, -0.5),   # 左下角
    #     1,                          # 宽度
    #     nrows,                      # 高度
    #     fill=False,
    #     edgecolor="black",
    #     linewidth=2,
    #     linestyle=":",
    #     zorder=2,
    #     clip_on=False
    # )
    # ax.add_patch(rect_middle)

    # 3) 它们交汇的格子：crimson 高亮
    rect_cross = Rectangle(
        (middle_col - 0.5, bottom_row - 0.5),
        1,
        1,
        fill=False,              # 不填充
        edgecolor="crimson",     # 只要 crimson 框
        linewidth=3,
        zorder=4,
        clip_on=False
    )
    ax.add_patch(rect_cross)

    # 主刻度位置
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))

    # -------------------------
    # x轴：只在最下面一行显示 ticks 和 labels
    # -------------------------
    if i // 3 == 1:
        ax.set_xticklabels(x_labels, rotation=90)
        ax.tick_params(
            axis='x',
            which='major',
            bottom=True,
            top=False,
            labelbottom=True,
            length=4
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(
            axis='x',
            which='major',
            bottom=False,
            top=False,
            labelbottom=False,
            length=0
        )

    # -------------------------
    # y轴：只在最左边一列显示 ticks 和 labels
    # -------------------------
    if i % 3 == 0:
        ax.set_yticklabels(y_labels)
        ax.tick_params(
            axis='y',
            which='major',
            left=True,
            right=False,
            labelleft=True,
            length=4
        )
    else:
        ax.set_yticklabels([])
        ax.tick_params(
            axis='y',
            which='major',
            left=False,
            right=False,
            labelleft=False,
            length=0
        )

    # 画每个格子的边界线
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(
        which='minor',
        color='white',
        linestyle='-',
        linewidth=0.6,
        alpha=0.35
    )
    ax.tick_params(which='minor', bottom=False, left=False)

    # 边框稍微保留
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

# =========================================================
# 4. 整体横纵坐标标签
# =========================================================

fig.supxlabel(
    r'Ridge regularization parameter $\alpha$',
    y=0.01,
    fontsize=24
)

fig.supylabel(
    r'Noise amplitude $\sigma$',
    x=0.01,
    fontsize=24
)

# =========================================================
# colorbar
# =========================================================
cbar_ax = fig.add_axes([0.88, 0.3, 0.02, 0.5])

cbar = fig.colorbar(
    im,
    cax=cbar_ax
)

# colorbar 只显示这些刻度
cbar.set_ticks(np.arange(0.5, 1.01, 0.1))

cbar.ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f'{x:.1f}'.rstrip('0').rstrip('.'))
)

# labelpad 控制 AUC 与 colorbar 的距离
cbar.set_label(
    'AUC',
    rotation=90,
    fontsize=22,
    labelpad=15,
    fontfamily='Arial'
)

cbar.ax.tick_params(
    labelsize=15,
    width=1.0,
    length=4
)

cbar.outline.set_linewidth(1.0)

# =========================================================

plt.subplots_adjust(
    left=0.1,
    right=0.85,
    top=0.93,
    bottom=0.18,
    wspace=0.15,
    hspace=0.2
)

plt.savefig('../figures/SFIG9.pdf',format='pdf')