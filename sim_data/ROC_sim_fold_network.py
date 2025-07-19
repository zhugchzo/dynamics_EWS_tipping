# import python libraries
import networkx as nx
import numpy as np
import pandas as pd
import os

if not os.path.exists('../data/fold_network/ROC'):
    os.makedirs('../data/fold_network/ROC')

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
sigma = 0.01

n = 20
md = 0.5

#----------------------------------
# Simulate model
#----------------------------------

# Network

def scale_free(n,md,seed):

    np.random.seed(seed)

    m = n * md
    gamma_in = 2.5
    gamma_out = 2.5

    alpha_in = 1/(gamma_in-1)
    alpha_out = 1/(gamma_out-1)

    w_in = np.ones(n)
    w_out = np.ones(n)
    edges = list()

    for i in range(n):
        w_in[i] = 1 / (1 + i)**alpha_in
        w_out[i] = 1 / (1 + i)**alpha_out

    w_in = np.random.permutation(w_in)
    w_out = np.random.permutation(w_out)
    
    w_in = w_in / np.sum(w_in)
    w_out = w_out / np.sum(w_out)

    l = 0
    while l < m:

        s = np.random.choice(range(n), p=w_out)
        t = np.random.choice(range(n), p=w_in)

        if s != t:
            edge = (s, t)
            if edge not in edges:
                edges.append(edge)
                l += 1

    g = nx.DiGraph()
    g.add_edges_from(edges)
    mapping = dict(zip(g,range(0,len(g))))  # delete isolated nodes
    g = nx.relabel_nodes(g,mapping)   # renumber the nodes

    return g

# Model

def de_fun(x, A, r, k, b, N):

    return r*x*(1-x/k) - b*x**2/(x**2+1) + np.sum(A*(x.reshape(N,1)*x),axis=1)/(1+x)

def recov_fun(x, A, r, k, b, N):

    jac= np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if i == j:
                jac[i,j] = r - 2*r*x[i]/k - b*2*x[i]/(x[i]**2+1)**2 + A[i,:]@x/(1+x[i])**2
            else:
                jac[i,j] = A[i,j]*x[i]/(1+x[i])

    try:
        evals = np.linalg.eigvals(jac)
    except:
        return 1997.0209, 1997.0209, 1997.0209, 1997.0209  # if error; This particular number has no other significance, it's just the author's birthday

    evals = [lam for lam in evals]
    dom_eval = max(evals, key=lambda x: x.real)

    evals.remove(dom_eval)
    oth_eval = evals[0]

    rrate = dom_eval.real
    imag_dom_eval = dom_eval.imag

    real_oth_eval = oth_eval.real
    imag_oth_eval = oth_eval.imag
    
    return rrate, imag_dom_eval, real_oth_eval, imag_oth_eval

r = 1
k = 9
c = 1 # grazing rate

bl = 1
bh = 2.5 # bifurcation parameter high
c_rate = 1.5e-5 # bifurcation parameter change rate

sim_len = int((bh-bl)*dt/c_rate)
series_len = 1000

for random_seed in range(100):

    np.random.seed(random_seed)

    # Scale-free network
    g = scale_free(n=20, md=0.5, seed=random_seed)

    N = len(g)

    w = np.random.uniform(0.5,1,len(g.edges())) # link weight
    source,target = zip(*g.edges())
    weight_edges = zip(source,target,w)
    g.add_weighted_edges_from(weight_edges)

    nodes = g.number_of_nodes()
    edges = g.number_of_edges()

    #generate weight adjacency matrix
    A = nx.to_numpy_array(g) #element: out_link
    for i,j,w in g.edges(data=True):
        A[i,j] = w['weight']
    A = A.T #elemnt: in_link

    network_A = A.copy()
    network_A[network_A != 0] = 1

    x0 = np.random.rand(N) * 9  #init
    equix0 = np.random.rand(N) * 9  #init

    t = np.arange(t0,sim_len,dt)

    x = np.zeros((N,len(t)))
    x_null = np.zeros((N,len(t)))
    equix = np.zeros((N,len(t)))

    rrate = np.zeros(len(t))

    b = pd.Series(np.linspace(bl,bh,len(t)),index=t)

    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = (N,int(tburn/dt)))
    dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = (N,len(t)))

    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = x0 + de_fun(x0, A, r, k, b[0], N)*dt + dW_burn[:,i]
        equix0 = equix0 + de_fun(equix0, A, r, k, b[0], N)*dt

    rrate0 = recov_fun(equix0, A, r, k, b[0], N)[0]

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
        x[:,i+1] = x[:,i] + de_fun(x[:,i], A, r, k, b.iloc[i], N)*dt + dW[:,i]
        x_null[:,i+1] = x_null[:,i] + de_fun(x_null[:,i], A, r, k, b.iloc[0], N)*dt + dW[:,i]
        equix[:,i+1] = equix[:,i] + de_fun(equix[:,i], A, r, k, b.iloc[i], N)*dt

        rrate[i+1] = recov_fun(equix[:,i+1], A, r, k, b.iloc[i], N)[0]

        if rrate[i+1] == 1997.0209:
            print('rrate Nan appears')
            break

        if np.any(np.isnan(x[:,i+1])):
            print('sim Nan appears')
            break

        # Determine the tipping point by the recovery rate changing from negative to positive
        if rrate[i] < 0 and rrate[i+1] > 0:
            rrate_record.append(i+1)

    if len(rrate_record) != 0:

        trans_time = rrate_record[0] # the time of tipping point

        c= np.linspace(0,trans_time,series_len)
        c = list(map(int,c))

        # Store series data in a temporary DataFrame
        data = {'t':t,'b':b,'real_eval1': rrate}
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

        print('Simulation complete {}'.format(random_seed))

    else:
        print('Cant find rrate = 0')

    #------------------------------------
    # Export data 
    #-----------------------------------

    filepath = '../data/fold_network/ROC/fold_network_{}.csv'.format(random_seed)
    filepath_null = '../data/fold_network/ROC/fold_network_null_{}.csv'.format(random_seed)

    df_temp_c.to_csv(filepath,index=False)
    df_temp_null_c.to_csv(filepath_null,index=False)

    np.savetxt('../data/fold_network/ROC/network_{}.csv'.format(random_seed), network_A, delimiter=',', fmt='%.3f')
















