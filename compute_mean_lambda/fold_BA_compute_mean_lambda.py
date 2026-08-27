import pandas as pd
import numpy as np
import os

if not os.path.exists('../results/mean_lambda/partial'):
    os.makedirs('../results/mean_lambda/partial')

if not os.path.exists('../results/mean_lambda/singlev'):
    os.makedirs('../results/mean_lambda/singlev')

df_re_sum = pd.DataFrame()
df_esti_re_sum = pd.DataFrame()
df_esti_im_sum = pd.DataFrame()

df_esti_re_v1_sum = pd.DataFrame()
df_esti_re_v2_sum = pd.DataFrame()

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    for random_seed in range(20):

        df_true_re = pd.read_csv('../data/fold_BA/ROC/{}/fold_BA_{}.csv'.format(network_size,random_seed))[['real_eval1']].iloc[-500:]
        df_esti_re = pd.read_csv('../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))[['esti_rrate']]
        df_esti_im = pd.read_csv('../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))[['esti_imag_dom_eval']]

        df_esti_re_v1 = pd.read_csv('../results/ROC/fold_BA/ews_singlev/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))[['esti_rrate1']]
        df_esti_re_v2 = pd.read_csv('../results/ROC/fold_BA/ews_singlev/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))[['esti_rrate2']]

        df_re_sum = pd.concat([df_re_sum.reset_index(drop=True),df_true_re.reset_index(drop=True)], axis=1)
        df_esti_re_sum = pd.concat([df_esti_re_sum.reset_index(drop=True),df_esti_re.reset_index(drop=True)], axis=1)
        df_esti_im_sum = pd.concat([df_esti_im_sum.reset_index(drop=True),df_esti_im.reset_index(drop=True)], axis=1)

        df_esti_re_v1_sum = pd.concat([df_esti_re_v1_sum.reset_index(drop=True),df_esti_re_v1.reset_index(drop=True)], axis=1)
        df_esti_re_v2_sum = pd.concat([df_esti_re_v2_sum.reset_index(drop=True),df_esti_re_v2.reset_index(drop=True)], axis=1)

true_re_mean = df_re_sum.mean(axis=1).to_frame(name='true_re_mean')

# 按行均值
esti_re_mean = df_esti_re_sum.mean(axis=1)
esti_im_mean = df_esti_im_sum.mean(axis=1)

esti_re_v1_mean = df_esti_re_v1_sum.mean(axis=1)
esti_re_v2_mean = df_esti_re_v2_sum.mean(axis=1)

# 按行标准差
esti_re_std = df_esti_re_sum.std(axis=1, ddof=1)
esti_im_std = df_esti_im_sum.std(axis=1, ddof=1)

esti_re_v1_std = df_esti_re_v1_sum.std(axis=1, ddof=1)
esti_re_v2_std = df_esti_re_v2_sum.std(axis=1, ddof=1)

# 每行有效样本数
n_re = df_esti_re_sum.count(axis=1)
n_im = df_esti_im_sum.count(axis=1)

n_re_v1 = df_esti_re_v1_sum.count(axis=1)
n_re_v2 = df_esti_re_v2_sum.count(axis=1)

# 95% 置信区间半宽
esti_re_ci = 1.96 * esti_re_std / np.sqrt(n_re)
esti_im_ci = 1.96 * esti_im_std / np.sqrt(n_im)

esti_re_v1_ci = 1.96 * esti_re_v1_std / np.sqrt(n_re_v1)
esti_re_v2_ci = 1.96 * esti_re_v2_std / np.sqrt(n_re_v2)

# 整理成一个 DataFrame
df_esti_stats = pd.DataFrame({
    'esti_re_mean': esti_re_mean,
    'esti_re_ci_lower': esti_re_mean - esti_re_ci,
    'esti_re_ci_upper': esti_re_mean + esti_re_ci,
    'esti_im_mean': esti_im_mean,
    'esti_im_ci_lower': esti_im_mean - esti_im_ci,
    'esti_im_ci_upper': esti_im_mean + esti_im_ci
})

df_esti_sv_stats = pd.DataFrame({
    'esti_re_v1_mean': esti_re_v1_mean,
    'esti_re_v1_ci_lower': esti_re_v1_mean - esti_re_v1_ci,
    'esti_re_v1_ci_upper': esti_re_v1_mean + esti_re_v1_ci,
    'esti_re_v2_mean': esti_re_v2_mean,
    'esti_re_v2_ci_lower': esti_re_v2_mean - esti_re_v2_ci,
    'esti_re_v2_ci_upper': esti_re_v2_mean + esti_re_v2_ci
})

df_mean = pd.concat([true_re_mean, df_esti_stats], axis=1)
df_mean.insert(0, 'Time', range(len(df_mean)))

df_mean.to_csv('../results/mean_lambda/fold_BA_mean_lambda.csv', index=False)

df_mean_sv = pd.concat([true_re_mean, df_esti_sv_stats], axis=1)
df_mean_sv.insert(0, 'Time', range(len(df_mean)))

df_mean_sv.to_csv('../results/mean_lambda/singlev/fold_BA_mean_lambda.csv', index=False)

###########################################################################################

for node_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:

    df_esti_re_part_sum = pd.DataFrame()
    df_esti_im_part_sum = pd.DataFrame()

    for network_size in [20, 30, 40, 50, 60]:

        for random_seed in range(20):

            df_esti_part_re = pd.read_csv('../results/ROC/fold_BA/ews_partial/{}/{}/dynamics_fold_BA_{}.csv'.format(node_ratio,network_size,random_seed))[['esti_rrate']]
            df_esti_part_im = pd.read_csv('../results/ROC/fold_BA/ews_partial/{}/{}/dynamics_fold_BA_{}.csv'.format(node_ratio,network_size,random_seed))[['esti_imag_dom_eval']]

            df_esti_re_part_sum = pd.concat([df_esti_re_part_sum.reset_index(drop=True),df_esti_part_re.reset_index(drop=True)], axis=1)
            df_esti_im_part_sum = pd.concat([df_esti_im_part_sum.reset_index(drop=True),df_esti_part_im.reset_index(drop=True)], axis=1)

    esti_re_part_mean = df_esti_re_part_sum.mean(axis=1)
    esti_im_part_mean = df_esti_im_part_sum.mean(axis=1)

    esti_re_part_std = df_esti_re_part_sum.std(axis=1, ddof=1)
    esti_im_part_std = df_esti_im_part_sum.std(axis=1, ddof=1)

    n_re_part = df_esti_re_part_sum.count(axis=1)
    n_im_part = df_esti_im_part_sum.count(axis=1)

    esti_re_part_ci = 1.96 * esti_re_part_std / np.sqrt(n_re_part)
    esti_im_part_ci = 1.96 * esti_im_part_std / np.sqrt(n_im_part)

    df_esti_part_stats = pd.DataFrame({
        'esti_re_mean': esti_re_part_mean,
        'esti_re_ci_lower': esti_re_part_mean - esti_re_part_ci,
        'esti_re_ci_upper': esti_re_part_mean + esti_re_part_ci,
        'esti_im_mean': esti_im_part_mean,
        'esti_im_ci_lower': esti_im_part_mean - esti_im_part_ci,
        'esti_im_ci_upper': esti_im_part_mean + esti_im_part_ci
    })

    df_mean_part = pd.concat([true_re_mean, df_esti_part_stats], axis=1)
    df_mean_part.insert(0, 'Time', range(len(df_mean_part)))

    df_mean_part.to_csv('../results/mean_lambda/partial/fold_BA_mean_lambda_{}.csv'.format(node_ratio), index=False)

###########################################################################################

if not os.path.exists('../results/mean_lambda/transient/fold_BA'):
    os.makedirs('../results/mean_lambda/transient/fold_BA')

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    df_transient_re_sum = pd.DataFrame()
    df_transient_esti_re_sum = pd.DataFrame()

    for random_seed in range(20):

        df = pd.read_csv('../results/transient/fold_BA/{}/dynamics_fold_BA_{}.csv'.format(network_size, random_seed))

        true_re = df['real_rrate'].to_numpy()
        esti_re = df['esti_rrate'].to_numpy()

        # 归一化时间轴到 [0, 1]
        x_true = np.linspace(0, 1, len(true_re))
        x_esti = np.linspace(0, 1, len(esti_re))

        x_new = np.linspace(0, 1, 100)

        # 插值到统一的 100 个点
        true_re_interp = np.interp(x_new, x_true, true_re)
        esti_re_interp = np.interp(x_new, x_esti, esti_re)

        df_transient_re_sum[random_seed] = true_re_interp
        df_transient_esti_re_sum[random_seed] = esti_re_interp

    true_transient_re_mean = df_transient_re_sum.mean(axis=1).to_frame(name='true_re_mean')

    # 按行均值
    esti_transient_re_mean = df_transient_esti_re_sum.mean(axis=1)

    # 按行标准差
    esti_transient_re_std = (df_transient_esti_re_sum.std(axis=1, ddof=1).fillna(0))

    # 每行有效样本数
    n_transient_re = df_transient_esti_re_sum.count(axis=1)

    # 95% 置信区间半宽
    esti_transient_re_ci = 1.96 * esti_transient_re_std / np.sqrt(n_transient_re)

    # 整理成一个 DataFrame
    df_transient_esti_stats = pd.DataFrame({
        'esti_re_mean': esti_transient_re_mean,
        'esti_re_ci_lower': esti_transient_re_mean - esti_transient_re_ci,
        'esti_re_ci_upper': esti_transient_re_mean + esti_transient_re_ci
    })

    df_transient_mean = pd.concat([true_transient_re_mean, df_transient_esti_stats], axis=1)

    df_transient_mean.insert(0, 'Normalized time', range(len(df_transient_mean)))

    df_transient_mean.to_csv('../results/mean_lambda/transient/fold_BA/fold_BA_mean_lambda_{}.csv'.format(network_size), index=False)

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