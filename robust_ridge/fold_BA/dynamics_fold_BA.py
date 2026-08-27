import numpy as np
import pandas as pd
import os
from sympy import symbols, diff

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

for sigma in [0.01, 0.02, 0.03, 0.04, 0.05]:

    # ridge parameter for regression
    for ridge_param in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:

        if not os.path.exists('../../results/ridge/fold_BA/ews/{}/{}'.format(sigma,ridge_param)):
            os.makedirs('../../results/ridge/fold_BA/ews/{}/{}'.format(sigma,ridge_param))

        for random_seed in range(50):

            df_tseries = pd.read_csv('../../data/fold_BA/noise/{}/fold_BA_{}.csv'.format(sigma,random_seed))
            df_network = pd.read_csv('../../data/fold_BA/noise/{}/network_{}.csv'.format(sigma,random_seed), header=None)

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
            dcop = 3 # x_j, bx_j, x_ix_j
            # total size of feature vector: linear + nonlinear + couple
            dtot = dlin + dnonlin + dcop

            keep_col_tseries = ['Time','t_diff','B']

            for node in range(N):
                keep_col_tseries.append('traj_x{}'.format(node))
                
            new_f_tseries = df_tseries[keep_col_tseries]
            values_tseries = new_f_tseries.values

            tseries = np.array(values_tseries[:,0])
            b_tseries = np.array(values_tseries[:,2])
            x_tseries = np.array(values_tseries[:,-N:])

            dt = values_tseries[1,1]

            x_mean = np.mean(x_tseries, axis=1)

            ##
            ## Global Ridge
            ##

            estimate_rrate = list()
            estimate_imag_dom_eval = list()
            estimate_real_oth_eval = list()
            estimate_imag_oth_eval = list()

            W_out_matrix = np.zeros((test_length,dtot + cte))

            initial_theta = 0
            delta_theta = 1e-3

            i = 0

            for winstart in range(test_length):

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

                    sum_y = A[node,:] @ y[:, winstart:winstart+train_length]

                    out_train[dlin + cnt + cte: dlin + cnt + cte + 1, node*train_length:(node+1)*train_length] = sum_y
                    out_train[dlin + cnt + cte + 1: dlin + cnt + cte + 2, node*train_length:(node+1)*train_length] = sum_y * np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)
                    out_train[dlin + cnt + cte + 2: dlin + cnt + cte + 3, node*train_length:(node+1)*train_length] = sum_y * x[:d,node*length + winstart : node*length + winstart + train_length]

                    fx_t = np.concatenate([fx_t, (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])[0]/dt])       

                    # the colunm of out_train_t is (cte, x, b, x^2, bx, x^3, bx^2, x_j, bx_j, x_ix_j).T
                    out_train_t[:,node] = out_train[:,(node+1)*train_length-1]

                # ridge regression: train W_out to map out_train to x[t+1] - x[t]
                W_out = fx_t @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))

                W_out_matrix[i,:] = W_out
                
                for node in range(N):

                    # estimating recovery rate

                    x_t = out_train_t[:,node][1]
                    b_t = out_train_t[:,node][2]
                    sum_y_t = out_train_t[:,node][7]

                    for j in range(N):
                        if j != node:
                            jacobian[node,j] = W_out[-3] + W_out[-2]*b_t + W_out[-1]*x_t
                        else:
                            jacobian[node,j] = diff_f(W_out[:dlin + dnonlin + cte], x_t, b_t) + W_out[-1]*sum_y_t

                rrate, imag_dom_eval, real_oth_eval, imag_oth_eval = solve_eval(jacobian)

                estimate_rrate.append(rrate)
                estimate_imag_dom_eval.append(imag_dom_eval)
                estimate_real_oth_eval.append(real_oth_eval)
                estimate_imag_oth_eval.append(imag_oth_eval)

                i += 1

            dic_pred = {'Time':tseries[train_length - 1 : -1], 'b_param':b_tseries[train_length - 1 : -1],'traj_x':x_mean[train_length - 1 : -1],
                        'esti_rrate':estimate_rrate, 'esti_real_oth_eval':estimate_real_oth_eval, 
                        'esti_imag_dom_eval':estimate_imag_dom_eval, 'esti_imag_oth_eval':estimate_imag_oth_eval}
            
            polynomial = ['cte', 'x', 'b', 'x^2', 'bx', 'x^3', 'bx^2', 'x_j', 'bx_j', 'x_ix_j']
            for j in range(dtot + cte):
                dic_pred[polynomial[j]] = W_out_matrix[:,j]

            csv_out = pd.DataFrame(dic_pred)
            csv_out.to_csv('../../results/ridge/fold_BA/ews/{}/{}/dynamics_fold_BA_{}.csv'.format(sigma,ridge_param,random_seed),header = True)


for sigma in [0.01, 0.02, 0.03, 0.04, 0.05]:

    # ridge parameter for regression
    for ridge_param in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:

        for random_seed in range(50):

            df_tseries = pd.read_csv('../../data/fold_BA/noise/{}/fold_BA_null_{}.csv'.format(sigma,random_seed))
            df_network = pd.read_csv('../../data/fold_BA/noise/{}/network_{}.csv'.format(sigma,random_seed), header=None)

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
            dcop = 3 # x_j, bx_j, x_ix_j
            # total size of feature vector: linear + nonlinear + couple
            dtot = dlin + dnonlin + dcop

            keep_col_tseries = ['Time','t_diff']

            for node in range(N):
                keep_col_tseries.append('traj_x{}'.format(node))
                
            new_f_tseries = df_tseries[keep_col_tseries]
            values_tseries = new_f_tseries.values

            tseries = np.array(values_tseries[:,0])
            x_tseries = np.array(values_tseries[:,-N:])

            dt = values_tseries[1,1]

            ##
            ## Global Ridge
            ##

            estimate_rrate = list()

            i = 0

            for winstart in range(test_length):

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

                    sum_y = A[node,:] @ y[:, winstart:winstart+train_length]

                    out_train[dlin + cnt + cte: dlin + cnt + cte + 1, node*train_length:(node+1)*train_length] = sum_y
                    out_train[dlin + cnt + cte + 1: dlin + cnt + cte + 2, node*train_length:(node+1)*train_length] = sum_y * np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)
                    out_train[dlin + cnt + cte + 2: dlin + cnt + cte + 3, node*train_length:(node+1)*train_length] = sum_y * x[:d,node*length + winstart : node*length + winstart + train_length]

                    fx_t = np.concatenate([fx_t, (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])[0]/dt])       

                    # the colunm of out_train_t is (cte, x, b, x^2, bx, x^3, bx^2, x_j, bx_j, x_ix_j).T
                    out_train_t[:,node] = out_train[:,(node+1)*train_length-1]

                # ridge regression: train W_out to map out_train to x[t+1] - x[t]
                W_out = fx_t @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))
                
                for node in range(N):

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

                estimate_rrate.append(rrate)

                i += 1

            dic_pred = {'Time':tseries[train_length - 1 : -1], 'esti_rrate':estimate_rrate}

            csv_out = pd.DataFrame(dic_pred)
            csv_out.to_csv('../../results/ridge/fold_BA/ews/{}/{}/dynamics_fold_BA_null_{}.csv'.format(sigma,ridge_param,random_seed),header = True)