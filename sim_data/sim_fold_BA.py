# import python libraries
import networkx as nx
import numpy as np
import pandas as pd
import os

if not os.path.exists('../data/fold_BA'):
    os.makedirs('../data/fold_BA')

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
sigma = 0.01

#----------------------------------
# Simulate model
#----------------------------------

# Network

def ba_network(n, m0, m):
    # 创建一个初始的完全图（包含m0个节点）
    g = nx.complete_graph(m0)
    
    # 更新节点度数
    degrees = {node: m0 - 1 for node in g.nodes}
    
    # 添加新节点直到网络达到n个节点
    for new_node in range(m0, n):
        # 计算连接概率
        total_degree = sum(degrees.values())
        probs = np.array([degrees[node] / total_degree for node in g.nodes])
        
        # 根据概率选择连接的m个节点
        chosen_nodes = np.random.choice(list(g.nodes), size=m, p=probs)
        
        # 新节点与选择的节点建立连接
        for node in chosen_nodes:
            g.add_edge(new_node, node)
            degrees[new_node] = degrees.get(new_node, 0) + 1  # 更新新节点度数
            degrees[node] += 1  # 更新已连接节点的度数

    return g

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

K = 5
C = 1
D = 5
E = 0.9
H = 0.1

epsl = 1

Bl = 0
Bh = 0.15
c_rate = 0.15e-5 # bifurcation parameter change rate

sim_len = int((Bh-Bl)*dt/c_rate)
series_len = 1000

random_seed = 45

np.random.seed(random_seed)

# SF network
network_size = 20
m0 = 5
m = 2

g = ba_network(network_size, m0, m)

N = len(g)

#generate weight adjacency matrix
A = nx.to_numpy_array(g)
A = A.T # out degree -> in degree

network_A = A.copy()

x0 = np.random.rand(N) * 1e-2  #init
equix0 = np.random.rand(N) * 1e-2  #init

t = np.arange(t0,sim_len,dt)

x = np.zeros((N,len(t)))
equix = np.zeros((N,len(t)))

rrate = np.zeros(len(t))
imag_dom_eval = np.zeros(len(t))
real_oth_eval = np.zeros(len(t))
imag_oth_eval = np.zeros(len(t))

B = pd.Series(np.linspace(Bl,Bh,len(t)),index=t)

# Create brownian increments (s.d. sqrt(dt))
dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = (N,int(tburn/dt)))
dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = (N,len(t)))

# Run burn-in period on x0
for i in range(int(tburn/dt)):

    x0 = x0 + de_fun(x0, epsl, A, N, B[0], K, C, D, E, H)*dt + dW_burn[:,i]
    equix0 = equix0 + de_fun(equix0, epsl, A, N, B[0], K, C, D, E, H)*dt

rrate0 = recov_fun(equix0, epsl, A, N, K, C, D, E, H)[0]
imag_dom_eval0 = recov_fun(equix0, epsl, A, N, K, C, D, E, H)[1]
real_oth_eval0 = recov_fun(equix0, epsl, A, N, K, C, D, E, H)[2]
imag_oth_eval0 = recov_fun(equix0, epsl, A, N, K, C, D, E, H)[3]

if rrate0 == 1997.0209 or np.any(np.isnan(x0)):
    print('Nan appears')

# Initial condition post burn-in period
x[:,0] = x0
equix[:,0] = equix0

rrate[0] = rrate0
imag_dom_eval[0] = imag_dom_eval0
real_oth_eval[0] = real_oth_eval0
imag_oth_eval[0] = imag_oth_eval0

rrate_record = []

# Run simulation
for i in range(len(t)-1):

    x[:,i+1] = x[:,i] + de_fun(x[:,i], epsl, A, N, B.iloc[i], K, C, D, E, H)*dt + dW[:,i]
    equix[:,i+1] = equix[:,i] + de_fun(equix[:,i], epsl, A, N, B.iloc[i], K, C, D, E, H)*dt

    x[:, i+1] = np.clip(x[:,i+1], 0, None)
    equix[:, i+1] = np.clip(equix[:,i+1], 0, None)

    rrate[i+1] = recov_fun(equix[:,i+1], epsl, A, N, K, C, D, E, H)[0]
    imag_dom_eval[i+1] = recov_fun(equix[:,i+1], epsl, A, N, K, C, D, E, H)[1]
    real_oth_eval[i+1] = recov_fun(equix[:,i+1], epsl, A, N, K, C, D, E, H)[2]
    imag_oth_eval[i+1] = recov_fun(equix[:,i+1], epsl, A, N, K, C, D, E, H)[3]

    if rrate[i+1] == 1997.0209:
        print('rrate Nan appears')

    if np.any(np.isnan(x[:,i+1])):
        print('sim Nan appears')

    # Determine the tipping point by the recovery rate changing from negative to positive
    if rrate[i] < 0 and rrate[i+1] > 0:
        rrate_record.append(i+1)

if len(rrate_record) != 0:

    trans_time = rrate_record[0] # the time of tipping point

    c1= np.linspace(0,trans_time,series_len)
    c1 = list(map(int,c1))

    c2 = c1.copy()

    c0 = np.linspace(trans_time,len(t)-1,int((len(t)-1-trans_time)*series_len/trans_time))
    c0 = c0[1:]
    c0 = list(map(int,c0))

    c2.extend(c0)

    # Store series data in a temporary DataFrame
    data = {'t':t,'B':B,'real_eval1': rrate, 'imag_eval1': imag_dom_eval, 'real_eval2': real_oth_eval, 'imag_eval2': imag_oth_eval}
    data_null = {'t':t}

    for node in range(N):
        data['traj_x{}'.format(node)] = x[node,:]
        data['equix{}'.format(node)] = equix[node,:]

    data['traj_x'] = np.mean(x, axis=0)
    data['equix'] = np.mean(equix, axis=0)

    df_temp = pd.DataFrame(data)

    df_temp_c1 = df_temp.iloc[c1].copy()
    df_temp_c2 = df_temp.iloc[c2].copy()

    df_temp_c1['Time'] = range(len(df_temp_c1))
    df_temp_c2['Time'] = range(len(df_temp_c2))

    df_temp_c1['t_diff'] = df_temp_c1['t'].diff()
    df_temp_c2['t_diff'] = df_temp_c2['t'].diff()

    cols = ['Time'] + [col for col in df_temp_c1.columns if col != 'Time']

    df_temp_c1 = df_temp_c1[cols]
    df_temp_c2 = df_temp_c2[cols]

    print('Simulation {} complete'.format(random_seed))

    #------------------------------------
    # Export data 
    #-----------------------------------

    filepath_train = '../data/fold_BA/fold_BA_train.csv'
    filepath = '../data/fold_BA/fold_BA.csv'

    df_temp_c1.to_csv(filepath_train,index=False)
    df_temp_c2.to_csv(filepath,index=False)

    np.savetxt('../data/fold_BA/network.csv', network_A, delimiter=',', fmt='%.3f')

else:
    print('Cant find rrate = 0')

