import numpy as np
import pandas as pd
import os
from sympy import symbols, diff

if not os.path.exists('../results/sub_hopf'):
    os.makedirs('../results/sub_hopf')

def diff_f(coefficient, x_t, y_t):

    x, y = symbols('x y')
    polynomial = np.array([1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3])

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

def stable_exponent(coefficient_F, coefficient_G, u_t, v_t, imag_t):

    u,v, x,y  = symbols('u v x y')
    polynomial = np.array([1, u, v, u**2, u*v, v**2, u**3, u**2*v, u*v**2, v**3])

    polynomial_F = np.dot(coefficient_F,polynomial)

    dF_du = diff(polynomial_F, u)
    dF_dv = diff(polynomial_F, v)

    dF_du_value = dF_du.subs({u: u_t, v: v_t})
    dF_dv_value = dF_dv.subs({u: u_t, v: v_t})

    polynomial_new = np.array([1, (x+u_t), (-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t), (x+u_t)**2, (x+u_t)*(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t), (-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t)**2, (x+u_t)**3,
                                (x+u_t)**2*(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t), (x+u_t)*(-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t)**2, (-(dF_du_value*x+imag_t*y)/dF_dv_value+v_t)**3])
    
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
stable_length = 30

train_length = 500 # Length of the rolling window
test_length = length - train_length

# choose cte = 1 to include constant term, cte = 0 to exclude it
cte = 1
# input dimension of x
d = 2
# size of linear part of feature vector (x1,...,xn)
dlin = d
# size of nonlinear part of feature vector
dnonlin_2 = int(((dlin + 1)*dlin/2))
dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6))
dnonlin = dnonlin_2 + dnonlin_3
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

# ridge parameter for regression
ridge_param = 1e-5

df_tseries = pd.read_csv('../data/sub_hopf/sub_hopf_train.csv')
keep_col_tseries = ['traj_x','traj_y','Time','real_eval1','imag_eval1','equix','equiy','t_diff']
new_f_tseries = df_tseries[keep_col_tseries]
values_tseries = new_f_tseries.values

x_tseries = np.array(values_tseries[:,:2])
tseries = np.array(values_tseries[:,2])
rrate_tseries = np.array(values_tseries[:,3])
imag_eval_tseries = np.array(values_tseries[:,4])
equix_tseries = np.array(values_tseries[:,-3:-1])

dt = values_tseries[1,-1]

##
## Ridge
##

# create an array to hold the linear part of the feature vector
x = np.zeros((dlin,length))

# fill in the linear part of the feature vector for all times
for j in range(length):
    x[0:d,j]=x_tseries[j]

esti_traj_x = list()
esti_traj_y = list()

estimate_rrate = list()
estimate_imag_dom_eval = list()
estimate_real_oth_eval = list()
estimate_imag_oth_eval = list()

estimate_stable = list()

W_out_matrix_x = np.zeros((test_length,dtot + cte))
W_out_matrix_y = np.zeros((test_length,dtot + cte))

i = 0

for winstart in range(test_length):

    # create an array to hold the full feature vector for training time
    # (use ones so the constant term is already 1)
    out_train = np.ones((dtot + cte,train_length))  

    # copy over the linear part (shift over by one to account for constant if needed)
    out_train[cte:dlin + cte,:] = x[:,winstart:winstart + train_length]

    # fill in the non-linear part, order = 2. x^2, xy, y^2
    cnt = 0
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            # shift by one for constant if needed
            out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
            cnt += 1

    # fill in the non-linear part, order = 3. x^3, x^2y, xy^2, y^3
    for row1 in range(dlin):
        for row2 in range(row1,dlin):
            for row3 in range(row2,dlin):
                out_train[dlin + cnt + cte] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                cnt += 1
      
    # the colunm of out_train_t is (cte, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3).T
    out_train_t = out_train[:,-1]

    # ridge regression: train W_out to map out_train to x[t+1] - x[t]
    W_out = (x[0:d,winstart + 1:winstart + train_length + 1] - x[0:d,winstart:winstart + train_length])/dt @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot + cte))
    
    W_out_matrix_x[i,:] = W_out[0,:]
    W_out_matrix_y[i,:] = W_out[1,:]
    #print(W_out)

    delta_x = (W_out @ out_train_t)*dt
    next_x = x[0:d,winstart + train_length - 1] + delta_x

    esti_traj_x.append(next_x[0])
    esti_traj_y.append(next_x[1])

    # estimating recovery rate

    x_t = out_train_t[1]
    y_t = out_train_t[2]

    dfx_dx = diff_f(W_out_matrix_x[i,:], x_t, y_t)[0]
    dfx_dy = diff_f(W_out_matrix_x[i,:], x_t, y_t)[1]
    dfy_dx = diff_f(W_out_matrix_y[i,:], x_t, y_t)[0]
    dfy_dy = diff_f(W_out_matrix_y[i,:], x_t, y_t)[1]

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

        stable = stable_exponent(W_out_matrix_x[i,:], W_out_matrix_y[i,:], x_t, y_t, abs(imag_dom_eval))

        estimate_stable.append(stable)

    i += 1

dic_pred = {'Time':tseries[train_length - 1 : -1],'traj_x':x_tseries[train_length - 1 : -1,0],'traj_y':x_tseries[train_length - 1 : -1,1],
            'equi_x':equix_tseries[train_length - 1 : -1,0],'equi_y':equix_tseries[train_length - 1 : -1,1],
            'next_x':x_tseries[train_length:,0],'next_y':x_tseries[train_length:,1],'pred_x':esti_traj_x, 'pred_y':esti_traj_y,
            'real_eval':rrate_tseries[train_length - 1 : -1],'esti_real_eval1':estimate_rrate, 'esti_real_eval2':estimate_real_oth_eval,
            'imag_eval':imag_eval_tseries[train_length - 1 : -1],'esti_imag_eval1':estimate_imag_dom_eval, 'esti_imag_eval2':estimate_imag_oth_eval}

f1_elements = ['cte_f1', 'x_f1', 'y_f1', 'x^2_f1', 'xy_f1', 'y^2_f1', 'x^3_f1', 'x^2y_f1', 'xy^2_f1', 'y^3_f1']
f2_elements = ['cte_f2', 'x_f2', 'y_f2', 'x^2_f2', 'xy_f2', 'y^2_f2', 'x^3_f2', 'x^2y_f2', 'xy^2_f2', 'y^3_f2']

for j in range(dtot + cte):
    dic_pred[f1_elements[j]] = W_out_matrix_x[:,j]
    dic_pred[f2_elements[j]] = W_out_matrix_y[:,j]

csv_out = pd.DataFrame(dic_pred)
csv_out.to_csv('../results/sub_hopf/sub_hopf_nopars.csv',header = True)

dic_stable = {'Time':tseries[-stable_length:], 'stable_expnt':estimate_stable}

csv_stable = pd.DataFrame(dic_stable)
csv_stable.to_csv('../results/sub_hopf/sub_hopf_stable_nopars.csv',header = True)
