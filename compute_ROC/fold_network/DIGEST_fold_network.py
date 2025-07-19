import numpy as np
import pandas as pd
import os

if not os.path.exists('../../results/ROC/fold_network/ews'):
    os.makedirs('../../results/ROC/fold_network/ews')

def diff_f(coefficient, x_t, b_t):

    polynomial = np.array([0, 1, 0, 2*x_t, b_t, 3*x_t**2, 2*b_t*x_t])

    df_dx_value = np.dot(coefficient,polynomial)

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

# for random_seed in range(100):

#     df_tseries = pd.read_csv('../../data/fold_network/ROC/fold_network_{}.csv'.format(random_seed))
#     df_network = pd.read_csv('../../data/fold_network/ROC/network_{}.csv'.format(random_seed), header=None)

#     N = len(df_network)
#     A = df_network.values

#     # choose cte = 1 to include constant term, cte = 0 to exclude it
#     cte = 1
#     # input dimension of x
#     d = 1
#     # size of linear part of feature vector (x1,...,xn,b)
#     dlin = d + 1
#     # size of nonlinear part of feature vector
#     dnonlin_2 = int(((dlin + 1)*dlin/2) - 1)
#     dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6) - (d+1))
#     dnonlin = dnonlin_2 + dnonlin_3
#     dcop = N*N
#     # total size of feature vector: linear + nonlinear + couple
#     dtot = dlin + dnonlin + dcop

#     # ridge parameter for regression
#     ridge_param = 1e-5

#     keep_col_tseries = ['Time','t_diff']

#     for node in range(N):
#         keep_col_tseries.append('traj_x{}'.format(node))
        
#     new_f_tseries = df_tseries[keep_col_tseries]
#     values_tseries = new_f_tseries.values

#     tseries = np.array(values_tseries[:,0])
#     x_tseries = np.array(values_tseries[:,-N:])

#     dt = values_tseries[1,1]

#     ##
#     ## Ridge
#     ##

#     estimate_rrate = list()

#     W_out_matrix = np.zeros((N,test_length,dlin + dnonlin + N + cte))

#     grid_initial_theta = [-2, -1, 0, 1, 2]
#     grid_delta_theta = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

#     i = 0

#     for winstart in range(test_length):

#         dic_AIC = {}

#         for initial_theta in grid_initial_theta:

#             for delta_theta in grid_delta_theta:

#                 # create an array to hold the linear part of the feature vector
#                 x = np.zeros((dlin,length*N))
#                 y = np.zeros((d*N,length))

#                 # create an array to hold the full feature vector for training time
#                 # (use ones so the constant term is already 1)
#                 out_train = np.zeros((dtot + cte,train_length*N))
#                 out_train[0, :] = 1

#                 for j in range(length):
#                     y[:,j] = x_tseries[j]

#                 W_out = np.zeros((N,cte + dlin + dnonlin + N))

#                 mse = 0

#                 for node in range(N):
#                     # fill in the linear part of the feature vector for all times
#                     for j in range(length):
#                         x[0:d,j+node*length] = x_tseries[j,node]
#                     x[-1,node*length + winstart : node*length + winstart + train_length] = np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)

#                     # copy over the linear part (shift over by one to account for constant if needed)
#                     out_train[cte:dlin + cte, node*train_length:(node+1)*train_length] = x[:,node*length + winstart : node*length + winstart + train_length]

#                     # fill in the non-linear part, order = 2
#                     cnt = 0
#                     for row1 in range(dlin):
#                         if row1 < dlin-1:
#                             for row2 in range(row1,dlin):
#                                 # shift by one for constant if needed
#                                 out_train[dlin + cnt + cte, node*train_length:(node+1)*train_length] = x[row1,node*length + winstart : node*length + winstart + train_length] * x[row2,node*length + winstart : node*length + winstart + train_length]
#                                 cnt += 1

#                     # fill in the non-linear part, order = 3
#                     for row1 in range(dlin):
#                         if row1 < dlin-1:
#                             for row2 in range(row1,dlin):
#                                 if row2 < dlin-1:
#                                     for row3 in range(row2,dlin):
#                                         out_train[dlin + cnt + cte, node*train_length:(node+1)*train_length] = x[row1,node*length + winstart : node*length + winstart + train_length] * x[row2,node*length + winstart : node*length + winstart + train_length] * x[row3,node*length + winstart : node*length + winstart + train_length]
#                                         cnt += 1

#                     out_train[dlin + cnt + cte + node*N : dlin + cnt + cte + (node+1)*N, node*train_length:(node+1)*train_length] = A[node,:].reshape(N,1) * y[:,winstart : winstart + train_length]

#                     out_train_node = np.vstack([out_train[:cte + dlin + dnonlin, node*train_length : (node+1)*train_length], out_train[cte + dlin + dnonlin + node*N : cte + dlin + dnonlin + (node+1)*N, node*train_length : (node+1)*train_length]])

#                     # ridge regression: train W_out to map out_train to x[t+1] - x[t]
#                     W_out_node = (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])/dt @ out_train_node[:,:].T @ np.linalg.pinv(out_train_node[:,:] @ out_train_node[:,:].T + ridge_param*np.identity(dlin + dnonlin + N + cte))

#                     # compute MSE
#                     mse += np.sum(((x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])/dt - W_out_node[0] @ out_train_node[:,:])**2)

#                     W_out[node,:] = W_out_node

#                 # compute AIC
#                 AIC = train_length*np.log(mse) + 2*np.count_nonzero(W_out)

#                 if np.isnan(AIC):
#                     AIC = 1e5

#                 dic_AIC['({},{})'.format(initial_theta,delta_theta)] = AIC

#         # find the best theta

#         formatted = {}

#         for key, value in dic_AIC.items():
#             row, col = key.strip('()').split(',')
#             formatted[(row, col)] = value

#         min_key = min(dic_AIC, key=dic_AIC.get)

#         best_initial_theta, best_delta_theta = min_key.strip('()').split(',')

#         best_initial_theta = float(best_initial_theta)
#         best_delta_theta = float(best_delta_theta)

#         # create an array to hold the linear part of the feature vector
#         x = np.zeros((dlin,length*N))
#         y = np.zeros((d*N,length))

#         jacobian = np.zeros((N,N))

#         # create an array to hold the full feature vector for training time
#         # (use ones so the constant term is already 1)
#         out_train = np.zeros((dtot + cte,train_length*N))
#         out_train[0, :] = 1

#         for j in range(length):
#             y[:,j] = x_tseries[j]

#         W_out = np.zeros((N,dtot + cte))

#         mse = 0

#         for node in range(N):
#             # fill in the linear part of the feature vector for all times
#             for j in range(length):
#                 x[0:d,j+node*length] = x_tseries[j,node]
#             x[-1,node*length + winstart : node*length + winstart + train_length] = np.linspace(best_initial_theta, best_initial_theta + (train_length - 1) * best_delta_theta, train_length)

#             # copy over the linear part (shift over by one to account for constant if needed)
#             out_train[cte:dlin + cte, node*train_length:(node+1)*train_length] = x[:,node*length + winstart : node*length + winstart + train_length]

#             # fill in the non-linear part, order = 2
#             cnt = 0
#             for row1 in range(dlin):
#                 if row1 < dlin-1:
#                     for row2 in range(row1,dlin):
#                         # shift by one for constant if needed
#                         out_train[dlin + cnt + cte, node*train_length:(node+1)*train_length] = x[row1,node*length + winstart : node*length + winstart + train_length] * x[row2,node*length + winstart : node*length + winstart + train_length]
#                         cnt += 1

#             # fill in the non-linear part, order = 3
#             for row1 in range(dlin):
#                 if row1 < dlin-1:
#                     for row2 in range(row1,dlin):
#                         if row2 < dlin-1:
#                             for row3 in range(row2,dlin):
#                                 out_train[dlin + cnt + cte, node*train_length:(node+1)*train_length] = x[row1,node*length + winstart : node*length + winstart + train_length] * x[row2,node*length + winstart : node*length + winstart + train_length] * x[row3,node*length + winstart : node*length + winstart + train_length]
#                                 cnt += 1

#             out_train[dlin + cnt + cte + node*N : dlin + cnt + cte + (node+1)*N, node*train_length:(node+1)*train_length] = A[node,:].reshape(N,1) * y[:,winstart : winstart + train_length]

#             out_train_node = np.vstack([out_train[:cte + dlin + dnonlin, node*train_length : (node+1)*train_length], out_train[cte + dlin + dnonlin + node*N : cte + dlin + dnonlin + (node+1)*N, node*train_length : (node+1)*train_length]])

#             # ridge regression: train W_out to map out_train to x[t+1] - x[t]
#             W_out_node = (x[0:d,node*length + winstart + 1:node*length + winstart + train_length + 1] - x[0:d,node*length + winstart:node*length + winstart + train_length])/dt @ out_train_node[:,:].T @ np.linalg.pinv(out_train_node[:,:] @ out_train_node[:,:].T + ridge_param*np.identity(dlin + dnonlin + N + cte))

#             W_out_matrix[node,i,:] = W_out_node[0]

#             # the colunm of out_train_t is (cte, x, b, x^2, bx, x^3, bx^2, couple_x).T
#             out_train_t = out_train_node[:,-1]

#             # estimating recovery rate

#             x_t = out_train_t[1]
#             b_t = out_train_t[2]

#             jacobian[node,:] = W_out_node[0][dlin + dnonlin + cte:]
#             jacobian[node,node] = jacobian[node,node] + diff_f(W_out_node[0][:dlin + dnonlin + cte], x_t, b_t)

#         rrate = solve_eval(jacobian)[0]

#         estimate_rrate.append(rrate)

#         i += 1

#     dic_pred = {'Time':tseries[train_length - 1 : -1], 'esti_rrate':estimate_rrate}

#     csv_out = pd.DataFrame(dic_pred)
#     csv_out.to_csv('../../results/ROC/fold_network/ews/DIGEST_fold_network_{}.csv'.format(random_seed),header = True)

for random_seed in range(100):

    df_tseries = pd.read_csv('../../data/fold_network/ROC/fold_network_null_{}.csv'.format(random_seed))
    df_network = pd.read_csv('../../data/fold_network/ROC/network_{}.csv'.format(random_seed), header=None)

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

    keep_col_tseries = ['Time','t_diff']

    for node in range(N):
        keep_col_tseries.append('traj_x{}'.format(node))
        
    new_f_tseries = df_tseries[keep_col_tseries]
    values_tseries = new_f_tseries.values

    tseries = np.array(values_tseries[:,0])
    x_tseries = np.array(values_tseries[:,-N:])

    dt = values_tseries[1,1]

    ##
    ## Ridge
    ##

    estimate_rrate = list()

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

            # estimating recovery rate

            x_t = out_train_t[1]
            b_t = out_train_t[2]

            jacobian[node,:] = W_out_node[0][dlin + dnonlin + cte:]
            jacobian[node,node] = jacobian[node,node] + diff_f(W_out_node[0][:dlin + dnonlin + cte], x_t, b_t)

        rrate = solve_eval(jacobian)[0]

        estimate_rrate.append(rrate)

        i += 1

    dic_pred = {'Time':tseries[train_length - 1 : -1], 'esti_rrate':estimate_rrate}

    csv_out = pd.DataFrame(dic_pred)
    csv_out.to_csv('../../results/ROC/fold_network/ews/DIGEST_fold_network_null_{}.csv'.format(random_seed),header = True)
