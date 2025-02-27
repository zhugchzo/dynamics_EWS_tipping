import numpy as np
import pandas as pd
import os
from sympy import symbols, diff

if not os.path.exists('../results/fold_network'):
    os.makedirs('../results/fold_network')

def diff_f(coefficient, x_t, b_t):

    x = symbols('x')
    polynomial = np.array([1, x, b_t, x**2, b_t*x, x**3, b_t*x**2])

    polynomial_f = np.dot(coefficient,polynomial)

    df_dx = diff(polynomial_f, x)

    df_dx_value = df_dx.subs({x: x_t})

    return df_dx_value

def solve_eval(jac):

    evals = np.linalg.eigvals(jac)

    evals = [lam for lam in evals]
    dom_eval = max(evals, key=lambda x: x.real)

    evals.remove(dom_eval)
    oth_eval = evals[0]

    rrate = dom_eval.real
    imag_dom_eval = dom_eval.imag

    real_oth_eval = oth_eval.real
    imag_oth_eval = oth_eval.imag
    
    return rrate, imag_dom_eval, real_oth_eval, imag_oth_eval

length = 1000

train_length = 500 # Length of the rolling window
test_length = length - train_length

df_tseries = pd.read_csv('../data/fold_network/fold_network_train.csv')
df_network = pd.read_csv('../data/fold_network/network.csv', header=None)

N = len(df_network)
A = df_network.values

# choose cte = 1 to include constant term, cte = 0 to exclude it
cte = 1
# input dimension of x
d = 1
# size of linear part of feature vector (x1,...,xn,b)
dlin = d + 1
# size of nonlinear part of feature vector
dnonlin_2 = int(((dlin + 1)*dlin/2) - 1)
dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6) - (d+1))
dnonlin = dnonlin_2 + dnonlin_3
dcop = N*N
# total size of feature vector: linear + nonlinear + couple
dtot = dlin + dnonlin + dcop

# ridge parameter for regression
ridge_param = 1e-5

keep_col_tseries = ['b','Time','real_eval1','imag_eval1','t_diff']

for node in range(N):
    keep_col_tseries.append('traj_x{}'.format(node))
for node in range(N):
    keep_col_tseries.append('equix{}'.format(node))
    
new_f_tseries = df_tseries[keep_col_tseries]
values_tseries = new_f_tseries.values

b_tseries = np.array(values_tseries[:,0])
tseries = np.array(values_tseries[:,1])
rrate_tseries = np.array(values_tseries[:,2])
imag_tseries = np.array(values_tseries[:,3])
x_tseries = np.array(values_tseries[:,-2*N:-N])
equix_tseries = np.array(values_tseries[:,-N:])

dt = values_tseries[1,4]

x_mean = np.mean(x_tseries, axis=1)
equix_mean = np.mean(equix_tseries, axis=1)

##
## Ridge
##

pred_traj_x = np.zeros((N,test_length))

estimate_rrate = list()
estimate_imag_dom_eval = list()
estimate_real_oth_eval = list()
estimate_imag_oth_eval = list()

best_initial_theta_list = list()
best_delta_theta_list = list()

W_out_matrix = np.zeros((N,test_length,dlin + dnonlin + N + cte))

grid_initial_theta = [-2, -1, 0, 1, 2]
grid_delta_theta = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

i = 0

for winstart in range(test_length):

    dic_AIC = {}

    for initial_theta in grid_initial_theta:

        for delta_theta in grid_delta_theta:

            # create an array to hold the linear part of the feature vector
            x = np.zeros((dlin,length*N))
            y = np.zeros((d*N,length))

            # create an array to hold the full feature vector for training time
            # (use ones so the constant term is already 1)
            out_train = np.zeros((dtot + cte,train_length*N))
            out_train[0, :] = 1

            for j in range(length):
                y[:,j] = x_tseries[j]

            W_out = np.zeros((N,cte + dlin + dnonlin + N))

            mse = 0

            for node in range(N):
                # fill in the linear part of the feature vector for all times
                for j in range(length):
                    x[0:d,j+node*length] = x_tseries[j,node]
                x[-1,node*length + winstart : node*length + winstart + train_length] = np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)

                # copy over the linear part (shift over by one to account for constant if needed)
                out_train[cte:dlin + cte, node*train_length:(node+1)*train_length] = x[:,node*length + winstart : node*length + winstart + train_length]

                # fill in the non-linear part, order = 2
                cnt = 0
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            # shift by one for constant if needed
                            out_train[dlin + cnt + cte, node*train_length:(node+1)*train_length] = x[row1,node*length + winstart : node*length + winstart + train_length] * x[row2,node*length + winstart : node*length + winstart + train_length]
                            cnt += 1

                # fill in the non-linear part, order = 3
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            if row2 < dlin-1:
                                for row3 in range(row2,dlin):
                                    out_train[dlin + cnt + cte, node*train_length:(node+1)*train_length] = x[row1,node*length + winstart : node*length + winstart + train_length] * x[row2,node*length + winstart : node*length + winstart + train_length] * x[row3,node*length + winstart : node*length + winstart + train_length]
                                    cnt += 1

                out_train[dlin + cnt + cte + node*N : dlin + cnt + cte + (node+1)*N, node*train_length:(node+1)*train_length] = A[node,:].reshape(N,1) * y[:,winstart : winstart + train_length]

                out_train_node = np.vstack([out_train[:cte + dlin + dnonlin, node*train_length : (node+1)*train_length], out_train[cte + dlin + dnonlin + node*N : cte + dlin + dnonlin + (node+1)*N, node*train_length : (node+1)*train_length]])

                # ridge regression: train W_out to map out_train to x[t+1] - x[t]
                W_out_node = (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])/dt @ out_train_node[:,:].T @ np.linalg.pinv(out_train_node[:,:] @ out_train_node[:,:].T + ridge_param*np.identity(dlin + dnonlin + N + cte))

                # compute MSE
                mse += np.sum(((x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])/dt - W_out_node[0] @ out_train_node[:,:])**2)

                W_out[node,:] = W_out_node

            # compute AIC
            AIC = train_length*np.log(mse) + 2*np.count_nonzero(W_out)

            if np.isnan(AIC):
                AIC = 1e5

            dic_AIC['({},{})'.format(initial_theta,delta_theta)] = AIC

    # find the best theta

    formatted = {}

    for key, value in dic_AIC.items():
        row, col = key.strip('()').split(',')
        formatted[(row, col)] = value

    min_key = min(dic_AIC, key=dic_AIC.get)

    best_initial_theta, best_delta_theta = min_key.strip('()').split(',')

    best_initial_theta = float(best_initial_theta)
    best_delta_theta = float(best_delta_theta)

    best_initial_theta_list.append(best_initial_theta)
    best_delta_theta_list.append(best_delta_theta)

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin,length*N))
    y = np.zeros((d*N,length))

    jacobian = np.zeros((N,N))

    # create an array to hold the full feature vector for training time
    # (use ones so the constant term is already 1)
    out_train = np.zeros((dtot + cte,train_length*N))
    out_train[0, :] = 1

    for j in range(length):
        y[:,j] = x_tseries[j]

    W_out = np.zeros((N,dtot + cte))

    mse = 0

    for node in range(N):
        # fill in the linear part of the feature vector for all times
        for j in range(length):
            x[0:d,j+node*length] = x_tseries[j,node]
        x[-1,node*length + winstart : node*length + winstart + train_length] = np.linspace(best_initial_theta, best_initial_theta + (train_length - 1) * best_delta_theta, train_length)

        # copy over the linear part (shift over by one to account for constant if needed)
        out_train[cte:dlin + cte, node*train_length:(node+1)*train_length] = x[:,node*length + winstart : node*length + winstart + train_length]

        # fill in the non-linear part, order = 2
        cnt = 0
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    # shift by one for constant if needed
                    out_train[dlin + cnt + cte, node*train_length:(node+1)*train_length] = x[row1,node*length + winstart : node*length + winstart + train_length] * x[row2,node*length + winstart : node*length + winstart + train_length]
                    cnt += 1

        # fill in the non-linear part, order = 3
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    if row2 < dlin-1:
                        for row3 in range(row2,dlin):
                            out_train[dlin + cnt + cte, node*train_length:(node+1)*train_length] = x[row1,node*length + winstart : node*length + winstart + train_length] * x[row2,node*length + winstart : node*length + winstart + train_length] * x[row3,node*length + winstart : node*length + winstart + train_length]
                            cnt += 1

        out_train[dlin + cnt + cte + node*N : dlin + cnt + cte + (node+1)*N, node*train_length:(node+1)*train_length] = A[node,:].reshape(N,1) * y[:,winstart : winstart + train_length]

        out_train_node = np.vstack([out_train[:cte + dlin + dnonlin, node*train_length : (node+1)*train_length], out_train[cte + dlin + dnonlin + node*N : cte + dlin + dnonlin + (node+1)*N, node*train_length : (node+1)*train_length]])

        # ridge regression: train W_out to map out_train to x[t+1] - x[t]
        W_out_node = (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])/dt @ out_train_node[:,:].T @ np.linalg.pinv(out_train_node[:,:] @ out_train_node[:,:].T + ridge_param*np.identity(dlin + dnonlin + N + cte))

        W_out_matrix[node,i,:] = W_out_node[0]

        # the colunm of out_train_t is (cte, x, b, x^2, bx, x^3, bx^2, couple_x).T
        out_train_t = out_train_node[:,-1]

        delta_x = (W_out_node @ out_train_t)*dt
        next_x = x[0:d,node*length + winstart + train_length - 1] + delta_x

        next_x = next_x[0]

        pred_traj_x[node,i] = next_x

        # estimating recovery rate

        x_t = out_train_t[1]
        b_t = out_train_t[2]

        jacobian[node,:] = W_out_node[0][dlin + dnonlin + cte:]
        jacobian[node,node] = jacobian[node,node] + diff_f(W_out_node[0][:dlin + dnonlin + cte], x_t, b_t)

    rrate = solve_eval(jacobian)[0]
    imag_dom_eval = solve_eval(jacobian)[1]
    real_oth_eval = solve_eval(jacobian)[2]
    imag_oth_eval = solve_eval(jacobian)[3]

    estimate_rrate.append(rrate)
    estimate_imag_dom_eval.append(imag_dom_eval)

    estimate_real_oth_eval.append(real_oth_eval)
    estimate_imag_oth_eval.append(imag_oth_eval)

    i += 1

pred_x_mean = np.mean(pred_traj_x, axis=0)

dic_pred = {'Time':tseries[train_length - 1 : -1],'b':b_tseries[train_length - 1 : -1],'traj_x':x_mean[train_length - 1 : -1],
            'equi_x':equix_mean[train_length - 1 : -1], 'next_x':x_mean[train_length:],'pred_x':pred_x_mean, 
            'real_eval':rrate_tseries[train_length - 1 : -1],'esti_real_eval1':estimate_rrate, 'esti_real_eval2':estimate_real_oth_eval,
            'imag_eval':imag_tseries[train_length - 1 : -1],'esti_imag_eval1':estimate_imag_dom_eval, 'esti_imag_eval2':estimate_imag_oth_eval,
            'initial_theta':best_initial_theta_list, 'delta_theta':best_delta_theta_list}

for node in range(N):

    dic_W_out = {'Time':tseries[train_length - 1 : -1]}
    W_out_node = np.hstack([W_out_matrix[node,:,0:dlin + dnonlin + node + cte], W_out_matrix[node,:,dlin + dnonlin + node + 1 + cte:]])

    polynomial = ['cte', 'x_{}'.format(node), 'b', 'x_{}^2'.format(node), 'bx_{}'.format(node), 'x_{}^3'.format(node), 'bx_{}^2'.format(node)]
    interaction = [i for i in range(N)]
    interaction.remove(node)
    
    for inter in interaction:
        polynomial.append('x_{}'.format(inter))

    for j in range(dlin + dnonlin + (N-1) + cte):
        dic_W_out[polynomial[j]] = W_out_node[:,j]
    
    csv_W_out = pd.DataFrame(dic_W_out)
    csv_W_out.to_csv('../results/fold_network/W_out_{}.csv'.format(node))

csv_out = pd.DataFrame(dic_pred)
csv_out.to_csv('../results/fold_network/fold_network.csv', header = True)
