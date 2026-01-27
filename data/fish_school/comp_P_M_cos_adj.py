import os
import pandas as pd
import numpy as np

# ==========================================================
# 工具函数
# ==========================================================
def unit_vector(x, y):
    '''将 (x, y) 归一化为单位向量；若为零向量则返回 NaN。'''
    norm = np.sqrt(x * x + y * y)
    with np.errstate(divide='ignore', invalid='ignore'):
        ux = np.where(norm > 0, x / norm, np.nan)
        uy = np.where(norm > 0, y / norm, np.nan)
    return ux, uy

def mean_pairwise_distance(xs, ys):
    '''计算一帧内所有个体的平均两两距离（只取上三角，忽略对角）。'''
    X = xs.values[:, None]
    Y = ys.values[:, None]
    dx = X - X.T
    dy = Y - Y.T
    D = np.sqrt(dx*dx + dy*dy)
    n = D.shape[0]
    if n <= 1:
        return np.nan
    iu = np.triu_indices(n, k=1)
    return float(np.mean(D[iu]))

def compute_adjacency_with_half(xs, ys, d_half):
    '''
    用半权距离 d_half 构造指数核邻接矩阵（对角置 0)
      w_ij = exp(-ln(2) * d_ij / d_half)
    当 d_ij = d_half 时, w_ij = 0.5
    '''
    X = xs.values[:, None]
    Y = ys.values[:, None]
    dx = X - X.T
    dy = Y - Y.T
    D = np.sqrt(dx*dx + dy*dy)

    if not np.isfinite(d_half) or d_half <= 0:
        W = np.zeros_like(D)
    else:
        W = np.exp(-np.log(2.0) * D / d_half)

    np.fill_diagonal(W, 0.0)
    return W

adjacency_dir = 'network'
os.makedirs(adjacency_dir, exist_ok=True)

input_file = 'fish_position.csv'

df = pd.read_csv(input_file)

df = df.sort_values(['id', 'imageNumber']).copy()
df['step'] = df.groupby('id').cumcount()

# 个体速度向量（头 - 尾）以及单位化
df['vx'] = df['xHead'] - df['xTail']
df['vy'] = df['yHead'] - df['yTail']
df['vhat_x'], df['vhat_y'] = unit_vector(df['vx'], df['vy'])

# ---------- 计算半权距离 d_half ----------
# 做法：对每个 step 计算一次平均两两距离，再对所有 step 取平均
step_means = []
for k in sorted(df['step'].unique()):
    blk = df[df['step'] == k]
    if len(blk) > 1:
        step_means.append(mean_pairwise_distance(blk['xBody'], blk['yBody']))
d_half = float(np.nanmean(step_means)) if len(step_means) > 0 else np.nan
# --------------------------------------------------------

records = []

initial = 2
train_length = 357
length = 394

steps = sorted(df['step'].unique())
for k in steps:

    block = df[df['step'] == k].dropna(subset=['vhat_x', 'vhat_y']).copy()
    if len(block) == 0:
        continue

    # 群体平均方向 -> 单位化得到 Vhat
    Vx_mean = block['vhat_x'].mean()
    Vy_mean = block['vhat_y'].mean()
    Vhat_x, Vhat_y = unit_vector(np.array([Vx_mean]), np.array([Vy_mean]))
    Vhat_x, Vhat_y = Vhat_x.item(), Vhat_y.item()

    # 每个体与群体方向的 cos
    cos_vals = block['vhat_x'] * Vhat_x + block['vhat_y'] * Vhat_y

    x_mean = block['xBody'].mean()
    y_mean = block['yBody'].mean()

    d_x = block['xBody'] - x_mean
    d_y = block['yBody'] - y_mean

    # v_x = block['vhat_x'] - Vx_mean
    # v_y = block['vhat_y'] - Vy_mean

    # theta = np.arctan2(d_y,d_x)
    # phi = np.arctan2(v_y,v_x)

    # M_vals = abs(np.mean(np.sin(theta-phi)))

    norm_d = np.sqrt(d_x**2 + d_y**2)
    norm_v = np.sqrt(block['vhat_x']**2 + block['vhat_y']**2)

    cross = (d_x * block['vhat_y'] - d_y * block['vhat_x']) / (norm_d * norm_v)

    M_vals = abs(np.mean(cross))

    row = {
        'step': int(k),
        'Time': block['Time'].iloc[0],
        'P':np.hypot(Vx_mean, Vy_mean),
        'M':float(M_vals.mean()),
        'Vhat_x':Vhat_x,
        'Vhat_y':Vhat_y
    }
    xh = block['xHead']
    yh = block['yHead']
    xt = block['xTail']
    yt = block['yTail']

    # 把每条鱼的 cos + 位置都加进去
    for fid, cosv, xh_i, yh_i, xt_i, yt_i in zip(
        block['id'].astype(int), cos_vals, xh, yh, xt, yt
    ):
        row[f'cos_{fid}'] = float(cosv)
        row[f'xHead_{fid}'] = float(xh_i)
        row[f'yHead_{fid}'] = float(yh_i)
        row[f'xTail_{fid}'] = float(xt_i)
        row[f'yTail_{fid}'] = float(yt_i)

    records.append(row)

    if initial <= k <= train_length:

        # 邻接矩阵（指数核，d_half = 平均两两距离）
        block_sorted = block.sort_values('id')
        xs = block_sorted['xBody']
        ys = block_sorted['yBody']
        ids_sorted = block_sorted['id'].astype(int).values
        W = compute_adjacency_with_half(xs, ys, d_half)
        adj_df = pd.DataFrame(W, index=ids_sorted, columns=ids_sorted)
        adj_df.to_csv(f'network/adj_{k-initial}.csv', float_format='%.6f')

if len(records) > 0:

    df_all = pd.DataFrame(records).sort_values('step').iloc[initial:length]
    df_all = df_all.drop(columns=['step'])

    df_train = df_all.head(train_length)

    df_all.to_csv('fish_school.csv', index=False)
    df_train.to_csv('fish_school_train.csv', index=False)