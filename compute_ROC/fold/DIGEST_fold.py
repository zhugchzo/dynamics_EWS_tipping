import numpy as np
import pandas as pd
import os
from sympy import symbols, diff

if not os.path.exists('../../results/ROC/fold/ews'):
    os.makedirs('../../results/ROC/fold/ews')

def diff_f(coefficient, x_t, b_t):

    x = symbols('x')
    polynomial = np.array([1, x, b_t, x**2, b_t*x, x**3, b_t*x**2])

    polynomial_f = np.dot(coefficient,polynomial)

    df_dx = diff(polynomial_f, x)

    df_dx_value = df_dx.subs({x: x_t})

    return df_dx_value

length = 1000

train_length = 500 # Length of the rolling window
test_length = length - train_length

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
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

# ridge parameter for regression
ridge_param = 1e-5

for random_seed in range(100):

    df_tseries = pd.read_csv('../../data/fold/ROC/fold_{}.csv'.format(random_seed))
    keep_col_tseries = ['traj_x','Time','t_diff']
    new_f_tseries = df_tseries[keep_col_tseries]
    values_tseries = new_f_tseries.values

    x_tseries = np.array(values_tseries[:,0])
    tseries = np.array(values_tseries[:,1])

    dt = values_tseries[1,-1]

    def rrate_update(W_out, partial_t):
        return W_out @ partial_t

    ##
    ## Ridge
    ##

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin,length))

    # fill in the linear part of the feature vector for all times
    for j in range(length):
        x[0:d,j]=x_tseries[j]

    estimate_rrate = list()
    best_initial_theta_list = list()
    best_delta_theta_list = list()

    W_out_matrix = np.zeros((test_length,dtot + cte))

    grid_initial_theta = [-2, -1, 0, 1, 2]
    grid_delta_theta = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    i = 0

    for winstart in range(test_length):
        
        dic_AIC = {}

        for initial_theta in grid_initial_theta:

            for delta_theta in grid_delta_theta:

                # create an array to hold the full feature vector for training time
                # (use ones so the constant term is already 1)
                out_train = np.ones((dtot + cte,train_length))  

                # copy over the linear part (shift over by one to account for constant if needed)
                x[-1,winstart:winstart + train_length] = np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)

                out_train[cte:dlin + cte,:] = x[:,winstart:winstart + train_length]

                # fill in the non-linear part, order = 2
                cnt = 0
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            # shift by one for constant if needed
                            out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
                            cnt += 1

                # fill in the non-linear part, order = 3
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            if row2 < dlin-1:
                                for row3 in range(row2,dlin):
                                    out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                                    cnt += 1

                # ridge regression: train W_out to map out_train to x[t+1] - x[t]
                W_out = (x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))
                
                # compute MSE
                mse = np.sum(((x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt - W_out[0] @ out_train[:,:])**2)

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

        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        out_train = np.ones((dtot + cte,train_length))  

        # copy over the linear part (shift over by one to account for constant if needed)
        x[-1,winstart:winstart + train_length] = np.linspace(best_initial_theta, best_initial_theta + (train_length - 1) * best_delta_theta, train_length)

        out_train[cte:dlin + cte,:] = x[:,winstart:winstart + train_length]

        # fill in the non-linear part, order = 2
        cnt = 0
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    # shift by one for constant if needed
                    out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
                    cnt += 1

        # fill in the non-linear part, order = 3
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    if row2 < dlin-1:
                        for row3 in range(row2,dlin):
                            out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                            cnt += 1

        # ridge regression: train W_out to map out_train to x[t+1] - x[t]
        W_out = (x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))

        W_out_matrix[i,:] = W_out[0]

        # the colunm of out_train_t is (cte, x, b, x^2, bx, x^3, bx^2).T
        out_train_t = out_train[:,-1]

        # estimating recovery rate

        x_t = out_train_t[1]
        b_t = out_train_t[2]

        rrate = diff_f(W_out_matrix[i,:], x_t, b_t)
        estimate_rrate.append(rrate)

        i += 1

    dic_pred = {'Time':tseries[train_length - 1 : -1],'esti_rrate':estimate_rrate}

    csv_out = pd.DataFrame(dic_pred)
    csv_out.to_csv('../../results/ROC/fold/ews/DIGEST_fold_{}.csv'.format(random_seed),header = True)

################################################################################################

for random_seed in range(100):

    df_tseries = pd.read_csv('../../data/fold/ROC/fold_null_{}.csv'.format(random_seed))
    keep_col_tseries = ['traj_x','Time','t_diff']
    new_f_tseries = df_tseries[keep_col_tseries]
    values_tseries = new_f_tseries.values

    x_tseries = np.array(values_tseries[:,0])
    tseries = np.array(values_tseries[:,1])

    dt = values_tseries[1,-1]

    def rrate_update(W_out, partial_t):
        return W_out @ partial_t

    ##
    ## Ridge
    ##

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin,length))

    # fill in the linear part of the feature vector for all times
    for j in range(length):
        x[0:d,j]=x_tseries[j]

    estimate_rrate = list()
    best_initial_theta_list = list()
    best_delta_theta_list = list()

    W_out_matrix = np.zeros((test_length,dtot + cte))

    grid_initial_theta = [-2, -1, 0, 1, 2]
    grid_delta_theta = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    i = 0

    for winstart in range(test_length):
        
        dic_AIC = {}

        for initial_theta in grid_initial_theta:

            for delta_theta in grid_delta_theta:

                # create an array to hold the full feature vector for training time
                # (use ones so the constant term is already 1)
                out_train = np.ones((dtot + cte,train_length))  

                # copy over the linear part (shift over by one to account for constant if needed)
                x[-1,winstart:winstart + train_length] = np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)

                out_train[cte:dlin + cte,:] = x[:,winstart:winstart + train_length]

                # fill in the non-linear part, order = 2
                cnt = 0
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            # shift by one for constant if needed
                            out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
                            cnt += 1

                # fill in the non-linear part, order = 3
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            if row2 < dlin-1:
                                for row3 in range(row2,dlin):
                                    out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                                    cnt += 1

                # ridge regression: train W_out to map out_train to x[t+1] - x[t]
                W_out = (x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))
                
                # compute MSE
                mse = np.sum(((x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt - W_out[0] @ out_train[:,:])**2)

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

        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        out_train = np.ones((dtot + cte,train_length))  

        # copy over the linear part (shift over by one to account for constant if needed)
        x[-1,winstart:winstart + train_length] = np.linspace(best_initial_theta, best_initial_theta + (train_length - 1) * best_delta_theta, train_length)

        out_train[cte:dlin + cte,:] = x[:,winstart:winstart + train_length]

        # fill in the non-linear part, order = 2
        cnt = 0
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    # shift by one for constant if needed
                    out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
                    cnt += 1

        # fill in the non-linear part, order = 3
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    if row2 < dlin-1:
                        for row3 in range(row2,dlin):
                            out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                            cnt += 1

        # ridge regression: train W_out to map out_train to x[t+1] - x[t]
        W_out = (x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))

        W_out_matrix[i,:] = W_out[0]

        # the colunm of out_train_t is (cte, x, b, x^2, bx, x^3, bx^2).T
        out_train_t = out_train[:,-1]

        # estimating recovery rate

        x_t = out_train_t[1]
        b_t = out_train_t[2]

        rrate = diff_f(W_out_matrix[i,:], x_t, b_t)
        estimate_rrate.append(rrate)

        i += 1

    dic_pred = {'Time':tseries[train_length - 1 : -1],'esti_rrate':estimate_rrate}

    csv_out = pd.DataFrame(dic_pred)
    csv_out.to_csv('../../results/ROC/fold/ews/DIGEST_fold_null_{}.csv'.format(random_seed),header = True)
