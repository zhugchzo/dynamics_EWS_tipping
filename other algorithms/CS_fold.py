import numpy as np
import pandas as pd
import os
from sympy import symbols, diff
import cvxpy as cp

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

if not os.path.exists('../results/fold'):
    os.makedirs('../results/fold')

def diff_f(coefficient, x_t, b_t):

    x = symbols('x')
    polynomial = np.array([x, b_t, x**2, b_t*x, x**3, b_t*x**2])

    polynomial_f = np.dot(coefficient,polynomial)

    df_dx = diff(polynomial_f, x)

    df_dx_value = df_dx.subs({x: x_t})

    return df_dx_value

length = 1000

train_length = 5 # Length of the rolling window
test_length = length - train_length

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

df_tseries = pd.read_csv('../data/fold/fold_train.csv')
keep_col_tseries = ['traj_x','b','Time','rrate']
new_f_tseries = df_tseries[keep_col_tseries]
values_tseries = new_f_tseries.values

x_tseries = np.array(values_tseries[:,0])
b_tseries = np.array(values_tseries[:,1])
tseries = np.array(values_tseries[:,2])
rrate_tseries = np.array(values_tseries[:,3])

def rrate_update(W_out, partial_t):
    return W_out @ partial_t

##
## NG-RC
##

# create an array to hold the linear part of the feature vector
x = np.zeros((dlin,length))

# fill in the linear part of the feature vector for all times
for j in range(length):
    x[0:d,j]=x_tseries[j]

predict_traj_x = list()
estimate_rrate = list()

W_out_matrix = np.zeros((test_length,dtot))

i = 0

for winstart in range(test_length):

    # create an array to hold the full feature vector for training time
    # (use ones so the constant term is already 1)
    out_train = np.ones((dtot,train_length))

    # copy over the linear part (shift over by one to account for constant if needed)
    x[-1,winstart:winstart + train_length] = np.arange(1, 1 + train_length * 0.01, 0.01)
    out_train[0:dlin,:] = x[:,winstart:winstart + train_length]

    # fill in the non-linear part, order = 2
    cnt = 0
    for row1 in range(dlin):
        if row1 < dlin-1:
            for row2 in range(row1,dlin):
                # shift by one for constant if needed
                out_train[dlin + cnt] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length]
                cnt += 1

    # fill in the non-linear part, order = 3
    for row1 in range(dlin):
        if row1 < dlin-1:
            for row2 in range(row1,dlin):
                if row2 < dlin-1:
                    for row3 in range(row2,dlin):
                        out_train[dlin + cnt] = x[row1,winstart:winstart + train_length] * x[row2,winstart:winstart + train_length] * x[row3,winstart:winstart + train_length]
                        cnt += 1
        
    # the colunm of out_train_t is (x, b, x^2, bx, x^3, bx^2).T
    out_train_t = out_train[:,-1]

    # Initialize the regression coefficient matrix W_out
    y = (x[0:d, winstart + 1:winstart + train_length + 1] - x[0:d, winstart:winstart + train_length]).T

    # 创建 cvxpy 变量
    W_out = cp.Variable(dtot)

    # 定义目标函数（最小化 L1 范数）
    objective = cp.Minimize(cp.norm(W_out, 1))

    # 定义约束条件
    constraints = [out_train_t.T @ W_out == y]

    # 定义优化问题
    problem = cp.Problem(objective, constraints)

    # 求解优化问题
    problem.solve(solver=cp.SCS)

    # 获取恢复的信号
    x_rec = W_out.value

    print(x_rec)

    W_out_matrix[i,:] = W_out

    delta_x = W_out @ out_train_t
    next_x = x[0:d,winstart + train_length-1] + delta_x

    next_x = next_x[0]

    predict_traj_x.append(next_x)

    # estimating recovery rate

    x_t = out_train_t[0]
    b_t = out_train_t[1]

    rrate = diff_f(W_out_matrix[i,:], x_t, b_t)
    estimate_rrate.append(rrate)

    i += 1

dic_pred = {'Time':tseries[train_length - 1 : -1],'b_param':b_tseries[train_length - 1 : -1],'next_x':x_tseries[train_length:],'pred_x':predict_traj_x,
            'rrate':rrate_tseries[train_length - 1 : -1],'esti_rrate':estimate_rrate}

polynomial = ['x', 'b', 'x^2', 'bx', 'x^3', 'bx^2']
for j in range(dtot):
    dic_pred[polynomial[j]] = W_out_matrix[:,j]

csv_out = pd.DataFrame(dic_pred)
csv_out.to_csv('../results/fold/CS_fold.csv',header = True)
