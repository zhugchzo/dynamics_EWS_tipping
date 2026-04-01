import pandas as pd
import numpy as np

df_re_sum = pd.DataFrame()
df_esti_re_sum = pd.DataFrame()
df_esti_im_sum = pd.DataFrame()

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(20):

        df_true_re = pd.read_csv('../data/fold_BA/ROC/{}/fold_BA_{}.csv'.format(network_size,random_seed))[['real_eval1']].iloc[-500:]
        df_esti_re = pd.read_csv('../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))[['esti_rrate']]
        df_esti_im = pd.read_csv('../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))[['esti_imag_dom_eval']]

        df_re_sum = pd.concat([df_re_sum.reset_index(drop=True),df_true_re.reset_index(drop=True)], axis=1)
        df_esti_re_sum = pd.concat([df_esti_re_sum.reset_index(drop=True),df_esti_re.reset_index(drop=True)], axis=1)
        df_esti_im_sum = pd.concat([df_esti_im_sum.reset_index(drop=True),df_esti_im.reset_index(drop=True)], axis=1)

true_re_mean = df_re_sum.mean(axis=1).to_frame(name='true_re_mean')

# 按行均值
esti_re_mean = df_esti_re_sum.mean(axis=1)
esti_im_mean = df_esti_im_sum.mean(axis=1)

# 按行标准差
esti_re_std = df_esti_re_sum.std(axis=1, ddof=1)
esti_im_std = df_esti_im_sum.std(axis=1, ddof=1)

# 每行有效样本数
n_re = df_esti_re_sum.count(axis=1)
n_im = df_esti_im_sum.count(axis=1)

# 95% 置信区间半宽
esti_re_ci = 1.96 * esti_re_std / np.sqrt(n_re)
esti_im_ci = 1.96 * esti_im_std / np.sqrt(n_im)

# 整理成一个 DataFrame
df_esti_stats = pd.DataFrame({
    'esti_re_mean': esti_re_mean,
    'esti_re_ci_lower': esti_re_mean - esti_re_ci,
    'esti_re_ci_upper': esti_re_mean + esti_re_ci,
    'esti_im_mean': esti_im_mean,
    'esti_im_ci_lower': esti_im_mean - esti_im_ci,
    'esti_im_ci_upper': esti_im_mean + esti_im_ci
})

df_mean = pd.concat([true_re_mean, df_esti_stats], axis=1)
df_mean.insert(0, 'Time', range(len(df_mean)))

df_mean.to_csv('../results/mean_lambda/fold_BA_mean_lambda.csv', index=False)

###########################################################################################

df_null_esti_sum = pd.DataFrame()

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(20):

        df_null_esti = pd.read_csv('../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_null_{}.csv'.format(network_size,random_seed))[['esti_rrate']]
        df_null_esti_sum = pd.concat([df_null_esti_sum.reset_index(drop=True),df_null_esti.reset_index(drop=True)], axis=1)

# 按行均值
esti_mean = df_null_esti_sum.mean(axis=1)

# 按行标准差
esti_std = df_null_esti_sum.std(axis=1, ddof=1)

# 每行有效样本数
n = df_null_esti_sum.count(axis=1)

# 95% 置信区间半宽
esti_ci = 1.96 * esti_std / np.sqrt(n)

# 整理成一个 DataFrame
df_null_esti_stats = pd.DataFrame({
    'esti_mean': esti_mean,
    'esti_ci_lower': esti_mean - esti_ci,
    'esti_ci_upper': esti_mean + esti_ci,
})

df_null_mean = pd.concat([df_null_esti_stats], axis=1)
df_null_mean.insert(0, 'Time', range(len(df_null_mean)))

df_null_mean.to_csv('../results/mean_lambda/fold_BA_null_mean_lambda.csv', index=False)