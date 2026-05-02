import pandas as pd
import numpy as np

all_bif_error = []
all_bif_type_value = []
time = None

for random_seed in range(200):

    file_path = '../results/analyze_bif/fold/bif/bif_out_{}.csv'.format(random_seed)

    df = pd.read_csv(file_path, na_values=['NA'])

    # 只从第一个 csv 取一次 time
    if time is None:
        time = df['Time']

    # bif_error: NA 不参与统计
    bif_error = pd.to_numeric(df['bif_error'], errors='coerce')
    all_bif_error.append(bif_error.rename('error_{}'.format(random_seed)))

    # bif_type_value: 全是 0/1，直接算均值
    bif_type_value = pd.to_numeric(df['bif_type_value'], errors='coerce')
    all_bif_type_value.append(bif_type_value.rename('type_{}'.format(random_seed)))

# 横向拼接
error_df = pd.concat(all_bif_error, axis=1)
type_df = pd.concat(all_bif_type_value, axis=1)

# bif_error 统计
bif_error_count = error_df.count(axis=1)

bif_error_mean = abs(error_df.mean(axis=1))

bif_error_std = error_df.std(axis=1, ddof=1)

bif_error_sem = bif_error_std / np.sqrt(bif_error_count)

bif_error_ci95 = 1.96 * bif_error_sem
bif_error_ci95[bif_error_count <= 1] = 0

bif_error_lower = bif_error_mean - bif_error_ci95
bif_error_upper = bif_error_mean + bif_error_ci95

# bif_type_value 均值
bif_type_value_mean = type_df.mean(axis=1)

stats_df = pd.DataFrame({
    'Time': time,
    'bif_error_mean': bif_error_mean,
    'bif_error_lower': bif_error_lower,
    'bif_error_upper': bif_error_upper,
    'count': bif_error_count,
    'bif_type_value_mean': bif_type_value_mean
})

stats_df.to_csv('../results/analyze_bif/fold/bif_output.csv', index=False)