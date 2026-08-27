import numpy as np
import pandas as pd
import os
from sympy import symbols, diff

if not os.path.exists('../../results/ROC/trans/ews'):
    os.makedirs('../../results/ROC/trans/ews')

def diff_f(coefficient, x_t, y_t, b_t):

    x, y = symbols('x y')
    polynomial = np.array([1, x, y, b_t, x**2, x*y, b_t*x, y**2, b_t*y, x**3, x**2*y, b_t*x**2, x*y**2, b_t*x*y, y**3, b_t*y**2])

    polynomial_f = np.dot(coefficient,polynomial)

    df_dx = diff(polynomial_f, x)
    df_dy = diff(polynomial_f, y)

    df_dx_value = df_dx.subs({x: x_t, y: y_t})
    df_dy_value = df_dy.subs({x: x_t, y: y_t})

    return df_dx_value, df_dy_value


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

# choose cte = 1 to include constant term, cte = 0 to exclude it
cte = 1
# input dimension of x
d = 2
# size of linear part of feature vector (x1,...,xn,b)
dlin = d + 1
# size of nonlinear part of feature vector
dnonlin_2 = int(((dlin + 1)*dlin/2) - 1)
dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6) - (d+1))
dnonlin = dnonlin_2 + dnonlin_3
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

# ridge parameter for regression
ridge_param = 1e-2

for random_seed in range(200):

    df_tseries = pd.read_csv('../../data/trans/ROC/trans_{}.csv'.format(random_seed))
    keep_col_tseries = ['traj_x','traj_y','Time','a','t_diff']
    new_f_tseries = df_tseries[keep_col_tseries]
    values_tseries = new_f_tseries.values

    x_tseries = np.array(values_tseries[:,:2])
    tseries = np.array(values_tseries[:,2])
    b_tseries = np.array(values_tseries[:,3])

    dt = values_tseries[1,-1]

    ##
    ## Ridge
    ##

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin,length))

    # fill in the linear part of the feature vector for all times
    for j in range(length):
        x[0:d,j]=x_tseries[j]

    estimate_rrate = list()
    estimate_imag_dom_eval = list()
    estimate_real_oth_eval = list()
    estimate_imag_oth_eval = list()

    best_initial_theta_list = list()
    best_delta_theta_list = list()

    W_out_matrix_x = np.zeros((test_length,dtot + cte))
    W_out_matrix_y = np.zeros((test_length,dtot + cte))

    grid_initial_theta = [-1, 0, 1]
    grid_delta_theta = [5e-4, 1e-3, 5e-3]

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

                # fill in the non-linear part, order = 2. x^2, xy, bx, y^2, by
                cnt = 0
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            # shift by one for constant if needed
                            out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
                            cnt += 1

                # fill in the non-linear part, order = 3. x^3, x^2y, bx^2, xy^2, bxy, y^3, by^2
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            if row2 < dlin-1:
                                for row3 in range(row2,dlin):
                                    out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                                    cnt += 1

                # compute the pseudo-inverse matrix error

                pinv_error_matrix = np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte)) @ (out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte)) - np.identity(out_train.shape[0])
                pinv_error = np.sum(pinv_error_matrix**2) / out_train.shape[0]**2   

                # ridge regression: train W_out to map out_train to x[t+1] - x[t]
                W_out = (x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))

    #####################################################################################

                # generation, compute MSE

                # create a place to store feature vectors for prediction
                out_gen = np.ones((dtot + cte))        # full feature vector
                x_gen = np.zeros((dlin,train_length))      # linear part

                x_gen[d:d+1,:] = np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)

                # copy over initial linear feature vector
                x_gen[:d,0] = x_tseries[winstart]

                gen_traj = np.zeros((d,train_length))
                gen_traj[:,0] = x_gen[0:d,0]

                for j in range(train_length-1):

                    # copy linear part into whole feature vector
                    out_gen[cte:dlin + cte] = x_gen[:,j] # shift by one for constant

                    # fill in the non-linear part

                    cnt = 0
                    # 2-order
                    for row1 in range(dlin):
                        if row1 < dlin-1:
                            for row2 in range(row1,dlin):
                                # shift by one for constant
                                out_gen[dlin + cnt + cte] = x_gen[row1,j] * x_gen[row2,j]
                                cnt += 1

                    # 3-order
                    for row1 in range(dlin):
                        if row1 < dlin-1:
                            for row2 in range(row1,dlin):
                                if row2 < dlin-1:
                                    for row3 in range(row2,dlin):
                                        # shift by one for constant
                                        out_gen[dlin + cnt + cte] = x_gen[row1,j] * x_gen[row2,j] * x_gen[row3,j]
                                        cnt += 1

                    # do a prediction
                    x_gen[0:d,j+1] = x_gen[0:d,j] + (W_out @ out_gen[:])*dt

                    gen_traj[:,j+1] = x_gen[0:d,j+1]

                sums = np.sum((x_tseries[winstart : winstart+train_length] - gen_traj.T)**2)
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

        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        out_train = np.ones((dtot + cte,train_length))  

        # copy over the linear part (shift over by one to account for constant if needed)
        x[-1,winstart:winstart + train_length] = np.linspace(best_initial_theta, best_initial_theta + (train_length - 1) * best_delta_theta, train_length)
        out_train[cte:dlin + cte,:] = x[:,winstart:winstart + train_length]

        # fill in the non-linear part, order = 2. x^2, xy, bx, y^2, by
        cnt = 0
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    # shift by one for constant if needed
                    out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
                    cnt += 1

        # fill in the non-linear part, order = 3. x^3, x^2y, bx^2, xy^2, bxy, y^3, by^2
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    if row2 < dlin-1:
                        for row3 in range(row2,dlin):
                            out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                            cnt += 1
            
        # the colunm of out_train_t is (cte, x, y, b, x^2, xy, bx, y^2, by, x^3, x^2y, bx^2, xy^2, bxy, y^3, by^2).T
        out_train_t = out_train[:,-1]

        # ridge regression: train W_out to map out_train to x[t+1] - x[t]
        W_out = (x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))
                
        W_out_matrix_x[i,:] = W_out[0,:]
        W_out_matrix_y[i,:] = W_out[1,:]

        # estimating recovery rate

        x_t = out_train_t[1]
        y_t = out_train_t[2]
        b_t = out_train_t[3]

        dfx_dx = diff_f(W_out_matrix_x[i,:], x_t, y_t, b_t)[0]
        dfx_dy = diff_f(W_out_matrix_x[i,:], x_t, y_t, b_t)[1]
        dfy_dx = diff_f(W_out_matrix_y[i,:], x_t, y_t, b_t)[0]
        dfy_dy = diff_f(W_out_matrix_y[i,:], x_t, y_t, b_t)[1]

        jacobian = np.array([[dfx_dx,dfx_dy],[dfy_dx,dfy_dy]], dtype=float)

        rrate, imag_dom_eval, real_oth_eval, imag_oth_eval = solve_eval(jacobian)

        estimate_rrate.append(rrate)
        estimate_imag_dom_eval.append(imag_dom_eval)
        estimate_real_oth_eval.append(real_oth_eval)
        estimate_imag_oth_eval.append(imag_oth_eval)

        i += 1

    dic_pred = {'Time':tseries[train_length - 1 : -1],'b':b_tseries[train_length - 1 : -1],
                'traj_x':x_tseries[train_length - 1 : -1,0],'traj_y':x_tseries[train_length - 1 : -1,1],
                'esti_rrate':estimate_rrate, 'esti_real_eval2':estimate_real_oth_eval,
                'esti_imag_eval1':estimate_imag_dom_eval, 'esti_imag_eval2':estimate_imag_oth_eval,
                'initial_theta':best_initial_theta_list, 'delta_theta':best_delta_theta_list}
    
    f1_elements = ['cte_f1', 'x_f1', 'y_f1', 'b_f1', 'x^2_f1', 'xy_f1', 'bx_f1', 'y^2_f1', 'by_f1', 'x^3_f1', 'x^2y_f1', 'bx^2_f1', 'xy^2_f1', 'bxy_f1', 'y^3_f1', 'by^2_f1']
    f2_elements = ['cte_f2', 'x_f2', 'y_f2', 'b_f2', 'x^2_f2', 'xy_f2', 'bx_f2', 'y^2_f2', 'by_f2', 'x^3_f2', 'x^2y_f2', 'bx^2_f2', 'xy^2_f2', 'bxy_f2', 'y^3_f2', 'by^2_f2']

    for j in range(dtot + cte):
        dic_pred[f1_elements[j]] = W_out_matrix_x[:,j]
        dic_pred[f2_elements[j]] = W_out_matrix_y[:,j]

    csv_out = pd.DataFrame(dic_pred)
    csv_out.to_csv('../../results/ROC/trans/ews/dynamics_trans_{}.csv'.format(random_seed),header = True)


for random_seed in range(200):

    df_tseries = pd.read_csv('../../data/trans/ROC/trans_null_{}.csv'.format(random_seed))
    keep_col_tseries = ['traj_x','traj_y','Time','t_diff']
    new_f_tseries = df_tseries[keep_col_tseries]
    values_tseries = new_f_tseries.values

    x_tseries = np.array(values_tseries[:,:2])
    tseries = np.array(values_tseries[:,2])

    dt = values_tseries[1,-1]

    ##
    ## Ridge
    ##

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin,length))

    # fill in the linear part of the feature vector for all times
    for j in range(length):
        x[0:d,j]=x_tseries[j]

    estimate_rrate = list()

    W_out_matrix_x = np.zeros((test_length,dtot + cte))
    W_out_matrix_y = np.zeros((test_length,dtot + cte))

    grid_initial_theta = [-1, 0, 1]
    grid_delta_theta = [5e-4, 1e-3, 5e-3]

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

                # fill in the non-linear part, order = 2. x^2, xy, bx, y^2, by
                cnt = 0
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            # shift by one for constant if needed
                            out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
                            cnt += 1

                # fill in the non-linear part, order = 3. x^3, x^2y, bx^2, xy^2, bxy, y^3, by^2
                for row1 in range(dlin):
                    if row1 < dlin-1:
                        for row2 in range(row1,dlin):
                            if row2 < dlin-1:
                                for row3 in range(row2,dlin):
                                    out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                                    cnt += 1

                # compute the pseudo-inverse matrix error

                pinv_error_matrix = np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte)) @ (out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte)) - np.identity(out_train.shape[0])
                pinv_error = np.sum(pinv_error_matrix**2) / out_train.shape[0]**2   

                # ridge regression: train W_out to map out_train to x[t+1] - x[t]
                W_out = (x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))

    #####################################################################################

                # generation, compute MSE

                # create a place to store feature vectors for prediction
                out_gen = np.ones((dtot + cte))        # full feature vector
                x_gen = np.zeros((dlin,train_length))      # linear part

                x_gen[d:d+1,:] = np.linspace(initial_theta, initial_theta + (train_length - 1) * delta_theta, train_length)

                # copy over initial linear feature vector
                x_gen[:d,0] = x_tseries[winstart]

                gen_traj = np.zeros((d,train_length))
                gen_traj[:,0] = x_gen[0:d,0]

                for j in range(train_length-1):

                    # copy linear part into whole feature vector
                    out_gen[cte:dlin + cte] = x_gen[:,j] # shift by one for constant

                    # fill in the non-linear part

                    cnt = 0
                    # 2-order
                    for row1 in range(dlin):
                        if row1 < dlin-1:
                            for row2 in range(row1,dlin):
                                # shift by one for constant
                                out_gen[dlin + cnt + cte] = x_gen[row1,j] * x_gen[row2,j]
                                cnt += 1

                    # 3-order
                    for row1 in range(dlin):
                        if row1 < dlin-1:
                            for row2 in range(row1,dlin):
                                if row2 < dlin-1:
                                    for row3 in range(row2,dlin):
                                        # shift by one for constant
                                        out_gen[dlin + cnt + cte] = x_gen[row1,j] * x_gen[row2,j] * x_gen[row3,j]
                                        cnt += 1

                    # do a prediction
                    x_gen[0:d,j+1] = x_gen[0:d,j] + (W_out @ out_gen[:])*dt

                    gen_traj[:,j+1] = x_gen[0:d,j+1]

                sums = np.sum((x_tseries[winstart : winstart+train_length] - gen_traj.T)**2)
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

        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        out_train = np.ones((dtot + cte,train_length))  

        # copy over the linear part (shift over by one to account for constant if needed)
        x[-1,winstart:winstart + train_length] = np.linspace(best_initial_theta, best_initial_theta + (train_length - 1) * best_delta_theta, train_length)
        out_train[cte:dlin + cte,:] = x[:,winstart:winstart + train_length]

        # fill in the non-linear part, order = 2. x^2, xy, bx, y^2, by
        cnt = 0
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    # shift by one for constant if needed
                    out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
                    cnt += 1

        # fill in the non-linear part, order = 3. x^3, x^2y, bx^2, xy^2, bxy, y^3, by^2
        for row1 in range(dlin):
            if row1 < dlin-1:
                for row2 in range(row1,dlin):
                    if row2 < dlin-1:
                        for row3 in range(row2,dlin):
                            out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                            cnt += 1
            
        # the colunm of out_train_t is (cte, x, y, b, x^2, xy, bx, y^2, by, x^3, x^2y, bx^2, xy^2, bxy, y^3, by^2).T
        out_train_t = out_train[:,-1]

        # ridge regression: train W_out to map out_train to x[t+1] - x[t]
        W_out = (x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))

        W_out_matrix_x[i,:] = W_out[0,:]
        W_out_matrix_y[i,:] = W_out[1,:]

        # estimating recovery rate

        x_t = out_train_t[1]
        y_t = out_train_t[2]
        b_t = out_train_t[3]

        dfx_dx = diff_f(W_out_matrix_x[i,:], x_t, y_t, b_t)[0]
        dfx_dy = diff_f(W_out_matrix_x[i,:], x_t, y_t, b_t)[1]
        dfy_dx = diff_f(W_out_matrix_y[i,:], x_t, y_t, b_t)[0]
        dfy_dy = diff_f(W_out_matrix_y[i,:], x_t, y_t, b_t)[1]

        jacobian = np.array([[dfx_dx,dfx_dy],[dfy_dx,dfy_dy]], dtype=float)

        rrate = solve_eval(jacobian)[0]

        estimate_rrate.append(rrate)

        i += 1

    dic_pred = {'Time':tseries[train_length - 1 : -1],'esti_rrate':estimate_rrate}

    csv_out = pd.DataFrame(dic_pred)
    csv_out.to_csv('../../results/ROC/trans/ews/dynamics_trans_null_{}.csv'.format(random_seed),header = True)