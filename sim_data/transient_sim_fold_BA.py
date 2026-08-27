# import python libraries
import networkx as nx
import numpy as np
import pandas as pd
import os

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
sigma = 0.01

#----------------------------------
# Simulate model
#----------------------------------

# Network

def ba_network(n,m0,m,direction='new_to_old',seed=None):

    if n < m0:
        raise ValueError('需要满足 n >= m0')

    if m0 < 2:
        raise ValueError('m0 至少为 2')

    if not 1 <= m <= m0:
        raise ValueError('需要满足 1 <= m <= m0')

    if direction not in {
        'new_to_old',
        'old_to_new',
        'random'
    }:
        raise ValueError(
            'direction 必须是 new_to_old、old_to_new 或 random'
        )

    rng = np.random.default_rng(seed)

    # 初始完全有向图：
    # 任意两个不同节点之间都有两个方向的边
    g = nx.DiGraph()
    g.add_nodes_from(range(m0))

    for u in range(m0):
        for v in range(m0):
            if u != v:
                g.add_edge(u, v)

    # 用总度数 = 入度 + 出度作为优先连接权重
    degrees = {
        node: g.in_degree(node) + g.out_degree(node)
        for node in g.nodes()
    }

    for new_node in range(m0, n):
        existing_nodes = np.array(
            list(g.nodes()),
            dtype=int
        )

        degree_values = np.array(
            [degrees[node] for node in existing_nodes],
            dtype=float
        )

        # 优先连接概率
        probabilities = degree_values / degree_values.sum()

        # 不重复选择，保证新节点连接 m 个不同旧节点
        chosen_nodes = rng.choice(
            existing_nodes,
            size=m,
            replace=False,
            p=probabilities
        )

        g.add_node(new_node)
        degrees[new_node] = 0

        for old_node in chosen_nodes:
            old_node = int(old_node)

            if direction == 'new_to_old':
                g.add_edge(new_node, old_node)

            elif direction == 'old_to_new':
                g.add_edge(old_node, new_node)

            else:
                if rng.random() < 0.5:
                    g.add_edge(new_node, old_node)
                else:
                    g.add_edge(old_node, new_node)

            # 每加入一条有向边，两个端点的总度数都增加 1
            degrees[new_node] += 1
            degrees[old_node] += 1

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

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    if not os.path.exists('../data/transient/fold_BA/{}'.format(network_size)):
        os.makedirs('../data/transient/fold_BA/{}'.format(network_size))

    for random_seed in range(20):

        np.random.seed(random_seed)

        # SF network
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

        B = pd.Series(np.linspace(Bl,Bh,len(t)),index=t)

        # Create brownian increments (s.d. sqrt(dt))
        dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = (N,int(tburn/dt)))
        dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = (N,len(t)))

        # Run burn-in period on x0
        for i in range(int(tburn/dt)):

            x0 = x0 + de_fun(x0, epsl, A, N, B[0], K, C, D, E, H)*dt + dW_burn[:,i]
            equix0 = equix0 + de_fun(equix0, epsl, A, N, B[0], K, C, D, E, H)*dt

        rrate0 = recov_fun(equix0, epsl, A, N, K, C, D, E, H)[0]

        if rrate0 == 1997.0209 or np.any(np.isnan(x0)):
            print('Nan appears')
        
        # Initial condition post burn-in period
        x[:,0] = x0
        equix[:,0] = equix0
        rrate[0] = rrate0

        rrate1_record = []
        rrate2_record = []
        relax_record = []

        # Run simulation
        for i in range(len(t)-1):

            x[:,i+1] = x[:,i] + de_fun(x[:,i], epsl, A, N, B.iloc[i], K, C, D, E, H)*dt + dW[:,i]
            equix[:,i+1] = equix[:,i] + de_fun(equix[:,i], epsl, A, N, B.iloc[i], K, C, D, E, H)*dt

            x[:, i+1] = np.clip(x[:,i+1], 0, None)
            equix[:, i+1] = np.clip(equix[:,i+1], 0, None)

            rrate[i+1] = recov_fun(equix[:,i+1], epsl, A, N, K, C, D, E, H)[0]

            if rrate[i+1] == 1997.0209:
                print('rrate Nan appears')

            if np.any(np.isnan(x[:,i+1])):
                print('sim Nan appears')

            # Determine the tipping point by the recovery rate changing from negative to positive
            if rrate[i] < 0 and rrate[i+1] > 0:
                rrate1_record.append(i+1)
            if rrate[i] > 0 and rrate[i+1] < 0: 
                rrate2_record.append(i+1)
            if (rrate2_record and i + 1 > rrate2_record[0] and abs(np.mean(equix[:, i+1]) - np.mean(equix[:, i])) < 1e-3):
                relax_record.append(i+50)

        if len(rrate1_record) != 0:

            trans_time = rrate1_record[0] # the time of tipping point
            relax_time = relax_record[0]

            # 第一阶段：0 到 trans_time，共 series_len 个采样点
            c1 = np.linspace(0, trans_time, series_len)

            # 第一阶段的采样间隔
            delta_t = trans_time / (series_len - 1)

            # 第二阶段所需的采样间隔数
            relax_intervals = round((relax_time - trans_time) / delta_t)

            # 第二阶段：避免重复包含 trans_time
            c0 = np.linspace(
                trans_time,
                relax_time,
                relax_intervals + 1
            )[1:]

            # 拼接完整时间序列
            c2 = np.concatenate((c1, c0))

            c2 = np.rint(c2).astype(int).tolist()

            # Store series data in a temporary DataFrame
            data = {'t':t,'B':B,'real_eval1': rrate}
            data_null = {'t':t}

            for node in range(N):
                data['traj_x{}'.format(node)] = x[node,:]
                data['equix{}'.format(node)] = equix[node,:]

            df_temp = pd.DataFrame(data)

            df_temp_c = df_temp.iloc[c2].copy()

            df_temp_c['Time'] = range(len(df_temp_c))

            df_temp_c['t_diff'] = df_temp_c['t'].diff()

            cols = ['Time'] + [col for col in df_temp_c.columns if col != 'Time']

            df_temp_c = df_temp_c[cols].iloc[500:]

            print('Network size = {}, Simulation {} complete'.format(network_size,random_seed))

            #------------------------------------
            # Export data 
            #-----------------------------------

            filepath = '../data/transient/fold_BA/{}/fold_BA_{}.csv'.format(network_size,random_seed)

            df_temp_c.to_csv(filepath,index=False)

        else:
            print('Cant find rrate = 0')