import numpy as np
import pandas as pd
import os

for node_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:

    for network_size in [20, 30, 40, 50, 60]:

        if not os.path.exists('../data/fold_ER/partial/{}/{}'.format(node_ratio,network_size)):
            os.makedirs('../data/fold_ER/partial/{}/{}'.format(node_ratio,network_size))

        for random_seed in range(20):

            df_tseries = pd.read_csv('../data/fold_ER/ROC/{}/fold_ER_{}.csv'.format(network_size,random_seed))
            df_null_tseries = pd.read_csv('../data/fold_ER/ROC/{}/fold_ER_null_{}.csv'.format(network_size,random_seed))
            df_network = pd.read_csv('../data/fold_ER/ROC/{}/network_{}.csv'.format(network_size,random_seed), header=None)

            N = len(df_network)
            A = df_network.values

            keep_col_tseries = ['Time','t_diff','real_eval1']

            rng = np.random.default_rng(random_seed)
            observe_node = np.sort(rng.choice(N, size=int(node_ratio*N), replace=False))

            for node in observe_node:
                keep_col_tseries.append('traj_x{}'.format(node))
                
            observe_df = df_tseries[keep_col_tseries]
            keep_col_tseries.remove('real_eval1')
            observe_df_null = df_null_tseries[keep_col_tseries]
            observe_A = A[np.ix_(observe_node, observe_node)]

            observe_df.to_csv('../data/fold_ER/partial/{}/{}/fold_ER_{}.csv'.format(node_ratio,network_size,random_seed))
            observe_df_null.to_csv('../data/fold_ER/partial/{}/{}/fold_ER_null_{}.csv'.format(node_ratio,network_size,random_seed))
            np.savetxt('../data/fold_ER/partial/{}/{}/network_{}.csv'.format(node_ratio,network_size,random_seed), observe_A, delimiter=',', fmt='%.3f')