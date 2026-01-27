# import python libraries
import networkx as nx
import numpy as np
import pandas as pd
import os

if not os.path.exists('../data/fold_ER/ROC'):
    os.makedirs('../data/fold_ER/ROC')

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
sigma = 0.01

#----------------------------------
# Simulate model
#----------------------------------


# Model

def de_fun(x, epsl, A, N, B, K, C, D, E, H):

    return B + x*(1-x/K)*(x/C-1) + epsl*np.sum(A*(x.reshape(N,1)*x/(D+E*x.reshape(N,1)+H*x)),axis=1)

def recov_fun(x, epsl, A, N, K, C, D, E, H):

    jac= np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if i == j:
                jac[i,j] = 2*x[i]/C + 2*x[i]/K - 3*x[i]**2/(C*K) - 1 + epsl*A[i,:]@((D*x+H*x**2)/(D+E*x[i]+H*x)**2)

            else:
                jac[i,j] = epsl*A[i,j]*(D*x[i]+E*x[i]**2)/((D+E*x[i]+H*x[j])**2)

    try:
        evals = np.linalg.eigvals(jac)
    except:
        return 1997.0209, 1997.0209, 1997.0209, 1997.0209 # if error; This particular number has no other significance, it's just the author's birthday

    evals = [lam for lam in evals]
    dom_eval = max(evals, key=lambda x: x.real)

    evals.remove(dom_eval)
    oth_eval = evals[0]

    rrate = dom_eval.real
    imag_dom_eval = dom_eval.imag

    real_oth_eval = oth_eval.real
    imag_oth_eval = oth_eval.imag
    
    return rrate, imag_dom_eval, real_oth_eval, imag_oth_eval

B = 0.1
K = 5
C = 1
D = 5
E = 0.9
H = 0.1

epsll = 0
epslh = 1
c_rate = 1e-5 # bifurcation parameter change rate

sim_len = int((epslh-epsll)*dt/c_rate)
series_len = 1000

for random_seed in range(100):

    np.random.seed(random_seed)

    # ER network
    network_size = 20
    link_p = 0.5

    g = nx.erdos_renyi_graph(n=network_size, p=link_p, directed=True, seed=int(random_seed))  # 使用示例参数生成网络

    N = len(g)

    #generate weight adjacency matrix
    A = nx.to_numpy_array(g)
    A = A.T # out degree -> in degree

    network_A = A.copy()

    x0 = np.random.rand(N) * 1e-2  #init
    equix0 = np.random.rand(N) * 1e-2  #init

    t = np.arange(t0,sim_len,dt)

    x = np.zeros((N,len(t)))
    x_null = np.zeros((N,len(t)))
    equix = np.zeros((N,len(t)))

    rrate = np.zeros(len(t))

    epsl = pd.Series(np.linspace(epsll,epslh,len(t)),index=t)

    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = (N,int(tburn/dt)))
    dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = (N,len(t)))

    # Run burn-in period on x0
    for i in range(int(tburn/dt)):

        x0 = x0 + de_fun(x0, epsl[0], A, N, B, K, C, D, E, H)*dt + dW_burn[:,i]
        equix0 = equix0 + de_fun(equix0, epsl[0], A, N, B, K, C, D, E, H)*dt

    rrate0 = recov_fun(equix0, epsl[0], A, N, K, C, D, E, H)[0]

    if rrate0 == 1997.0209 or np.any(np.isnan(x0)):
        print('Nan appears')
    
    # Initial condition post burn-in period
    x[:,0] = x0
    x_null[:,0] = x0
    equix[:,0] = equix0

    rrate[0] = rrate0

    rrate_record = []

    # Run simulation
    for i in range(len(t)-1):

        x[:,i+1] = x[:,i] + de_fun(x[:,i], epsl.iloc[i], A, N, B, K, C, D, E, H)*dt + dW[:,i]
        x_null[:,i+1] = x_null[:,i] + de_fun(x_null[:,i], epsl.iloc[0], A, N, B, K, C, D, E, H)*dt + dW[:,i]
        equix[:,i+1] = equix[:,i] + de_fun(equix[:,i], epsl.iloc[i], A, N, B, K, C, D, E, H)*dt

        x[:, i+1] = np.clip(x[:,i+1], 0, None)
        x_null[:, i+1] = np.clip(x_null[:,i+1], 0, None)
        equix[:, i+1] = np.clip(equix[:,i+1], 0, None)

        rrate[i+1] = recov_fun(equix[:,i+1], epsl.iloc[i+1], A, N, K, C, D, E, H)[0]

        if rrate[i+1] == 1997.0209:
            print('rrate Nan appears')

        if np.any(np.isnan(x[:,i+1])):
            print('sim Nan appears')

        # Determine the tipping point by the recovery rate changing from negative to positive
        if rrate[i] < 0 and rrate[i+1] > 0:
            rrate_record.append(i+1)

    if len(rrate_record) != 0:

        trans_time = rrate_record[0] # the time of tipping point

        c= np.linspace(0,trans_time,series_len)
        c = list(map(int,c))

        # Store series data in a temporary DataFrame
        data = {'t':t,'epsl':epsl,'real_eval1': rrate}
        data_null = {'t':t}

        for node in range(N):
            data['traj_x{}'.format(node)] = x[node,:]
            data['equix{}'.format(node)] = equix[node,:]

            data_null['traj_x{}'.format(node)] = x_null[node,:]

        df_temp = pd.DataFrame(data)
        df_temp_null = pd.DataFrame(data_null)

        df_temp_c = df_temp.iloc[c].copy()
        df_temp_null_c = df_temp_null.iloc[c].copy()

        df_temp_c['Time'] = range(len(df_temp_c))
        df_temp_null_c['Time'] = range(len(df_temp_null_c))

        df_temp_c['t_diff'] = df_temp_c['t'].diff()
        df_temp_null_c['t_diff'] = df_temp_null_c['t'].diff()

        cols = ['Time'] + [col for col in df_temp_c.columns if col != 'Time']
        cols_null = ['Time'] + [col for col in df_temp_null_c.columns if col != 'Time']

        df_temp_c = df_temp_c[cols]
        df_temp_null_c = df_temp_null_c[cols_null]

        print('Simulation {} complete'.format(random_seed))

        #------------------------------------
        # Export data 
        #-----------------------------------

        filepath = '../data/fold_ER/ROC/fold_ER_{}.csv'.format(random_seed)
        filepath_null = '../data/fold_ER/ROC/fold_ER_null_{}.csv'.format(random_seed)

        df_temp_c.to_csv(filepath,index=False)
        df_temp_null_c.to_csv(filepath_null,index=False)

        np.savetxt('../data/fold_ER/ROC/network_{}.csv'.format(random_seed), network_A, delimiter=',', fmt='%.3f')

    else:
        print('Cant find rrate = 0')

