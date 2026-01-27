import numpy as np
import pandas as pd
import os
from sympy import symbols, diff

if not os.path.exists('../../results/ROC/sub_hopf/ews'):
    os.makedirs('../../results/ROC/sub_hopf/ews')

if not os.path.exists('../../results/ROC/sub_hopf/stable'):
    os.makedirs('../../results/ROC/sub_hopf/stable')

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

def stable_exponent(coefficient_F, coefficient_G, u_t, v_t, b_t, imag_t):

    u,v, x,y  = symbols('u v x y')
    polynomial = np.array([1, u, v, b_t, u**2, u*v, b_t*u, v**2, b_t*v, u**3, u**2*v, b_t*u**2, u*v**2, b_t*u*v, v**3, b_t*v**2])

    polynomial_F = np.dot(coefficient_F,polynomial)

    dF_du = diff(polynomial_F, u)
    dF_dv = diff(polynomial_F, v)

    dF_du_value = dF_du.subs({u: u_t, v: v_t})
    dF_dv_value = dF_dv.subs({u: u_t, v: v_t})

    polynomial_new = np.array([1, (x+u_t), (-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t), b_t, (x+u_t)**2, (x+u_t)*(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t), b_t*(x+u_t),
                                (-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t)**2, b_t*(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t), (x+u_t)**3, (x+u_t)**2*(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t), 
                                b_t*(x+u_t)**2, (x+u_t)*(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t)**2, b_t*(x+u_t)*(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t),
                                  (-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t)**3, b_t*(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t)**2])
    
    polynomial_f = np.dot(coefficient_F,polynomial_new) + imag_t*y
    polynomial_g = -(dF_du_value*np.dot(coefficient_F,polynomial_new) + dF_dv_value*np.dot(coefficient_G,polynomial_new))/imag_t - imag_t*x

    df_dx = diff(polynomial_f, x)
    df_dy = diff(polynomial_f, y)

    df_dxdx = diff(df_dx, x)
    df_dxdy = diff(df_dx, y)
    df_dydy = diff(df_dy, y)

    df_dxdxdx = diff(df_dxdx, x)
    df_dxdydy = diff(df_dxdy, y)

    df_dxdx_value = df_dxdx.subs({x:0,y:0})
    df_dxdy_value = df_dxdy.subs({x:0,y:0})
    df_dydy_value = df_dydy.subs({x:0,y:0})

    df_dxdxdx_value = df_dxdxdx.subs({x:0,y:0})
    df_dxdydy_value = df_dxdydy.subs({x:0,y:0})

    dg_dx = diff(polynomial_g, x)
    dg_dy = diff(polynomial_g, y)

    dg_dxdx = diff(dg_dx, x)
    dg_dxdy = diff(dg_dx, y)
    dg_dydy = diff(dg_dy, y)

    dg_dxdxdy = diff(dg_dxdx, y)
    dg_dydydy = diff(dg_dydy, y)

    dg_dxdx_value = dg_dxdx.subs({x:0,y:0})
    dg_dxdy_value = dg_dxdy.subs({x:0,y:0})
    dg_dydy_value = dg_dydy.subs({x:0,y:0})

    dg_dxdxdy_value = dg_dxdxdy.subs({x:0,y:0})
    dg_dydydy_value = dg_dydydy.subs({x:0,y:0})

    stable = 1/16*(df_dxdxdx_value + df_dxdydy_value + dg_dxdxdy_value + dg_dydydy_value) + 1/(16*imag_t)*(df_dxdy_value*(df_dxdx_value + df_dydy_value) - dg_dxdy_value*(dg_dxdx_value + dg_dydy_value) - df_dxdx_value*dg_dxdx_value + df_dydy_value*dg_dydy_value)

    return stable

length = 1000

train_length = 500 # Length of the rolling window
test_length = length - train_length

stable_length = 30

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
ridge_param = 1e-5

for random_seed in range(100):

    df_tseries = pd.read_csv('../../data/sub_hopf/ROC/sub_hopf_{}.csv'.format(random_seed))
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
    estimate_imag_dom_eval = list()
    estimate_real_oth_eval = list()
    estimate_imag_oth_eval = list()

    estimate_stable = list()

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
        imag_dom_eval = solve_eval(jacobian)[1]
        real_oth_eval = solve_eval(jacobian)[2]
        imag_oth_eval = solve_eval(jacobian)[3]

        estimate_rrate.append(rrate)
        estimate_imag_dom_eval.append(imag_dom_eval)
        estimate_real_oth_eval.append(real_oth_eval)
        estimate_imag_oth_eval.append(imag_oth_eval)

        if winstart >= test_length - stable_length:

            stable = stable_exponent(W_out_matrix_x[i,:], W_out_matrix_y[i,:], x_t, y_t, b_t, abs(imag_dom_eval))

            estimate_stable.append(stable)

        i += 1

    dic_pred = {'Time':tseries[train_length - 1 : -1],'esti_rrate':estimate_rrate, 'esti_real_eval2':estimate_real_oth_eval
                ,'esti_imag_eval1':estimate_imag_dom_eval, 'esti_imag_eval2':estimate_imag_oth_eval}

    csv_out = pd.DataFrame(dic_pred)
    csv_out.to_csv('../../results/ROC/sub_hopf/ews/dynamics_sub_hopf_{}.csv'.format(random_seed),header = True)

    dic_stable = {'Time':tseries[-stable_length:], 'stable_expnt':estimate_stable}

    csv_stable = pd.DataFrame(dic_stable)
    csv_stable.to_csv('../../results/ROC/sub_hopf/stable/sub_hopf_stable{}.csv'.format(random_seed),header = True)

for random_seed in range(100):

    df_tseries = pd.read_csv('../../data/sub_hopf/ROC/sub_hopf_null_{}.csv'.format(random_seed))
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
    csv_out.to_csv('../../results/ROC/sub_hopf/ews/dynamics_sub_hopf_null_{}.csv'.format(random_seed),header = True)