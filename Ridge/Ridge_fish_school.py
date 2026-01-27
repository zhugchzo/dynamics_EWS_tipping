import numpy as np
import pandas as pd
import os
from sympy import symbols, diff

if not os.path.exists('../results/fish_school'):
    os.makedirs('../results/fish_school')

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

length = 357

train_length = 300 # Length of the rolling window
test_length = length - train_length

df_tseries = pd.read_csv('../data/fish_school/fish_school_train.csv')

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
dcop = 3 # x_j, bx_j, x_ix_j
# total size of feature vector: linear + nonlinear + couple
dtot = dlin + dnonlin + dcop

# ridge parameter for regression
ridge_param = 1e-5

N = 25
col_cos = ['cos_{}'.format(i) for i in range(N)]

tseries = df_tseries['Time'].to_numpy()
x_tseries = df_tseries[col_cos].to_numpy()

##
## Global Ridge
##

pred_traj_x = np.zeros((N,test_length))

estimate_rrate = list()
estimate_imag_dom_eval = list()
estimate_real_oth_eval = list()
estimate_imag_oth_eval = list()

best_initial_theta_list = list()
best_delta_theta_list = list()

grid_initial_theta = [-1, 0, 1]
grid_delta_theta = [5e-4, 1e-3, 5e-3]

i = 0

for winstart in range(test_length):

    adj_sum = []

    for network_i in range(winstart,winstart+train_length):

        df_network = pd.read_csv('../data/fish_school/network/adj_{}.csv'.format(network_i), header=None)
        
        A_name = df_network.values
        A_i = A_name[1:, 1:].astype(float)

        adj_sum.append(A_i)

    # sum_matrix = np.sum(adj_sum, axis=0)
    # A = sum_matrix / len(adj_sum)

    dic_AIC = {}

    for initial_theta in grid_initial_theta:

        for delta_theta in grid_delta_theta:

            # create an array to hold the linear part of the feature vector
            x = np.zeros((dlin,length*N))
            y = np.zeros((d*N,length))
            
            fx_t = np.array([])
            out_train_t = np.zeros((dtot + cte,N))
            jacobian = np.zeros((N,N))

            # create an array to hold the full feature vector for training time
            # (use ones so the constant term is already 1)
            out_train = np.zeros((dtot + cte,train_length*N))
            out_train[0, :] = 1

            for j in range(length):
                y[:,j] = x_tseries[j]

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

                A_node_sum = np.zeros((N,train_length))

                for network_j in range(train_length):

                    A = adj_sum[network_j]
                    A_node_sum[:,network_j] = A[node,:]

                sum_y = np.sum(A_node_sum * y[:, winstart:winstart+train_length],axis=0)

                out_train[dlin + cnt + cte: dlin + cnt + cte + 1, node*train_length:(node+1)*train_length] = sum_y
                out_train[dlin + cnt + cte + 1: dlin + cnt + cte + 2, node*train_length:(node+1)*train_length] = sum_y * np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)
                out_train[dlin + cnt + cte + 2: dlin + cnt + cte + 3, node*train_length:(node+1)*train_length] = sum_y * x[:d,node*length + winstart : node*length + winstart + train_length]

                # fx_t = np.concatenate([fx_t, (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])[0]/dt_tseries[winstart:winstart+train_length]])     
                fx_t = np.concatenate([fx_t, (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])[0]])

                # the colunm of out_train_t is (cte, x, b, x^2, bx, x^3, bx^2, x_j, bx_j, x_ix_j, x_j^2).T
                out_train_t[:,node] = out_train[:,(node+1)*train_length-1]

            # compute the pseudo-inverse matrix error

            pinv_error_matrix = np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte)) @ (out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte)) - np.identity(out_train.shape[0])
            pinv_error = np.sum(pinv_error_matrix**2) / out_train.shape[0]**2

            # ridge regression: train W_out to map out_train to x[t+1] - x[t]
            W_out = fx_t @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))

#####################################################################################

            # generation, compute MSE

            # create a place to store feature vectors for prediction
            out_gen = np.ones((dtot + cte,N))        # full feature vector
            x_gen = np.zeros((dlin + dcop,train_length*N))      # linear part

            x_gen[d:d+1,:] = np.tile(np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length), N)

            gen_traj = np.zeros((N,train_length))

            for node in range(N):

                x_gen[0,train_length*node] = x_tseries[winstart,node]

                sum_y0 = A[node,:] @ x_tseries[winstart]
            
                out_gen[dlin + cnt + cte: dlin + cnt + cte + 1, node] = sum_y0
                out_gen[dlin + cnt + cte + 1: dlin + cnt + cte + 2, node] = sum_y0 * x_gen[d:d+1,train_length*node]
                out_gen[dlin + cnt + cte + 2: dlin + cnt + cte + 3, node] = sum_y0 * x_gen[0,train_length*node]

                gen_traj[node,0] = x_gen[0,train_length*node]

            for j in range(train_length-1):

                for node in range(N):

                    # copy linear part into whole feature vector
                    out_gen[cte:dlin + cte,node] = x_gen[:dlin,train_length*node+j] # shift by one for constant

                    # fill in the non-linear part, order = 2
                    cnt = 0
                    for row1 in range(dlin):
                        if row1 < dlin-1:
                            for row2 in range(row1,dlin):
                                # shift by one for constant if needed
                                out_gen[dlin + cnt + cte, node] = x_gen[row1, train_length*node+j] * x[row2, train_length*node+j]
                                cnt += 1

                    # fill in the non-linear part, order = 3
                    for row1 in range(dlin):
                        if row1 < dlin-1:
                            for row2 in range(row1,dlin):
                                if row2 < dlin-1:
                                    for row3 in range(row2,dlin):
                                        out_gen[dlin + cnt + cte, node] = x[row1, train_length*node+j] * x[row2, train_length*node+j] * x[row3, train_length*node+j]
                                        cnt += 1

                    # do a prediction
                    x_gen[0,train_length*node+j+1] = W_out @ out_gen[:,node]

                    gen_traj[node,j+1] = x_gen[0,train_length*node+j+1]

            sums = np.sum((x_tseries[winstart : winstart+train_length].T - gen_traj)**2)

            mse = sums / train_length

            # compute AIC
            if np.log(mse) >= 0:
                AIC = pinv_error*train_length*np.log(mse) + 2*np.count_nonzero(W_out)
            else:
                AIC = train_length*np.log(mse)/pinv_error + 2*np.count_nonzero(W_out)

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
    
    fx_t = np.array([])
    out_train_t = np.zeros((dtot + cte,N))
    jacobian = np.zeros((N,N))

    # create an array to hold the full feature vector for training time
    # (use ones so the constant term is already 1)
    out_train = np.zeros((dtot + cte,train_length*N))
    out_train[0, :] = 1

    for j in range(length):
        y[:,j] = x_tseries[j]

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

        A_node_sum = np.zeros((N,train_length))

        for network_j in range(train_length):

            A = adj_sum[network_j]
            A_node_sum[:,network_j] = A[node,:]

        sum_y = np.sum(A_node_sum * y[:, winstart:winstart+train_length],axis=0)

        out_train[dlin + cnt + cte: dlin + cnt + cte + 1, node*train_length:(node+1)*train_length] = sum_y
        out_train[dlin + cnt + cte + 1: dlin + cnt + cte + 2, node*train_length:(node+1)*train_length] = sum_y * np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)
        out_train[dlin + cnt + cte + 2: dlin + cnt + cte + 3, node*train_length:(node+1)*train_length] = sum_y * x[:d,node*length + winstart : node*length + winstart + train_length]

        # fx_t = np.concatenate([fx_t, (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])[0]/dt_tseries[winstart:winstart+train_length]])
        fx_t = np.concatenate([fx_t, (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])[0]])
      
        # the colunm of out_train_t is (cte, x, b, x^2, bx, x^3, bx^2, x_j, bx_j, x_ix_j).T
        out_train_t[:,node] = out_train[:,(node+1)*train_length-1]

    # ridge regression: train W_out to map out_train to x[t+1] - x[t]
    W_out = fx_t @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))
    
    for node in range(N):

        # delta_x = (W_out @ out_train_t[:,node])*dt_tseries[winstart+train_length]
        # next_x = x[0:d,node*length + winstart + train_length - 1] + delta_x

        # pred_traj_x[node,i] = next_x[0]

        # estimating recovery rate

        x_t = out_train_t[:,node][1]
        b_t = out_train_t[:,node][2]
        sum_y_t = out_train_t[:,node][7]

        for j in range(N):
            if j != node:
                jacobian[node,j] = W_out[-3] + W_out[-2]*b_t + W_out[-1]*x_t
            else:
                jacobian[node,j] = diff_f(W_out[:dlin + dnonlin + cte], x_t, b_t) + W_out[-1]*sum_y_t

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

dic_pred = {'Time':tseries[train_length - 1 : -1],'esti_real_eval1':estimate_rrate, 'esti_real_eval2':estimate_real_oth_eval,
            'esti_imag_eval1':estimate_imag_dom_eval, 'esti_imag_eval2':estimate_imag_oth_eval,
            'initial_theta':best_initial_theta_list, 'delta_theta':best_delta_theta_list}

csv_out = pd.DataFrame(dic_pred)
csv_out.to_csv('../results/fish_school/fish_school.csv', header = True)