import pandas as pd
from scipy.stats import kendalltau

time_list = list()
tau_list = list()

df = pd.read_csv('../results/rotifers-algae/rotifers-algae.csv')

for i in range(10,len(df)+1):

    t = df['Time'].to_numpy()[:i].reshape(-1)
    rrate = df['esti_rrate'].to_numpy()[:i].reshape(-1)

    tau, p_value = kendalltau(t, rrate)

    time_list.append(t[-1])
    tau_list.append(tau)

    dic_tau = {'Time':time_list, 'tau':tau_list}

    csv_out = pd.DataFrame(dic_tau)

    csv_out.to_csv('../results/rotifers-algae/rotifers-algae_tau.csv',header = True)