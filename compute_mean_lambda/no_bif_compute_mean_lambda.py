import pandas as pd
import numpy as np

df_esti_sum = pd.DataFrame()

for random_seed in range(200):

    df_esti = pd.read_csv('../results/ROC/no_bif/ews/dynamics_no_bif_{}.csv'.format(random_seed))[['esti_rrate']]

    df_esti_sum = pd.concat([df_esti_sum.reset_index(drop=True),df_esti.reset_index(drop=True)], axis=1)

# 按行均值
esti_mean = df_esti_sum.mean(axis=1)

# 按行标准差
esti_std = df_esti_sum.std(axis=1, ddof=1)

# 每行有效样本数
n = df_esti_sum.count(axis=1)

# 95% 置信区间半宽
esti_ci = 1.96 * esti_std / np.sqrt(n)

# 整理成一个 DataFrame
df_esti_stats = pd.DataFrame({
    'esti_mean': esti_mean,
    'esti_ci_lower': esti_mean - esti_ci,
    'esti_ci_upper': esti_mean + esti_ci,
})

df_esti_stats.insert(0, 'Time', range(len(df_esti_stats)))

df_esti_stats.to_csv('../results/mean_lambda/no_bif_mean_lambda.csv', index=False)