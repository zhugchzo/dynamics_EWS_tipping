import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sympy import symbols, diff
import random

if not os.path.exists('../results/fold'):
    os.makedirs('../results/fold')

length = 1000

train_length = 500 # Length of the rolling window
test_length = length - train_length

# choose cte = 1 to include constant term, cte = 0 to exclude it
cte = 0
# input dimension of x
d = 1
# size of linear part of feature vector (x1,...,xn,b)
dlin = d + 1
# size of nonlinear part of feature vector
dnonlin_2 = int(((dlin + 1)*dlin/2) - 1)
dnonlin_3 = int(((dlin + 2)*(dlin + 1)*dlin/6) / 2)
dnonlin = dnonlin_2 + dnonlin_3
# total size of feature vector: linear + nonlinear
dtot = dlin + dnonlin

df_tseries = pd.read_csv('../data/fold/fold_train.csv')
keep_col_tseries = ['traj_x','b','Time','rrate','t_diff']
new_f_tseries = df_tseries[keep_col_tseries]
values_tseries = new_f_tseries.values

x_tseries = np.array(values_tseries[:,0])
b_tseries = np.array(values_tseries[:,1])
tseries = np.array(values_tseries[:,2])
rrate_tseries = np.array(values_tseries[:,3])

dt = values_tseries[1,-1]

def diff_f(coefficient, x_t, b_t):

    x = symbols('x')
    polynomial = np.array([x, b_t, x**2, b_t*x, x**3, b_t*x**2])

    polynomial_f = np.dot(coefficient,polynomial)

    df_dx = diff(polynomial_f, x)

    df_dx_value = df_dx.subs({x: x_t})

    return df_dx_value

rand_seed = 25

np.random.seed(rand_seed)
tf.random.set_seed(rand_seed)
random.seed(rand_seed)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(dtot,), kernel_initializer='lecun_uniform'),
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer='lecun_uniform'),
        tf.keras.layers.Dense(1, kernel_initializer='lecun_uniform')
    ])
    return model

model = create_model()

optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer_adam, loss='mean_squared_error')

initial_weights = model.get_weights()

# create an array to hold the linear part of the feature vector
x = np.zeros((dlin,length))

# fill in the linear part of the feature vector for all times
for j in range(length):
    x[0:d,j]=x_tseries[j]

x[dlin-1,:] = b_tseries
xb = x

predict_traj_x = list()
estimate_rrate = list()

W_out_matrix = np.zeros((test_length,dtot + cte))

i = 0 # count

for winstart in range(test_length):

    # create an array to hold the full feature vector for training time
    # (use ones so the constant term is already 1)
    out_train = np.ones((dtot + cte,train_length))  

    # copy over the linear part (shift over by one to account for constant if needed)
    out_train[cte:dlin + cte,:] = xb[:,winstart:winstart + train_length]

    # fill in the non-linear part, order = 2
    cnt = 0
    for row1 in range(dlin):
        if row1 < dlin-1:
            for row2 in range(row1,dlin):
                # shift by one for constant if needed
                out_train[dlin + cnt + cte] = xb[row1,winstart:winstart + train_length] * xb[row2,winstart:winstart + train_length]
                cnt += 1
        else:
            continue

    # fill in the non-linear part, order = 3
    for row1 in range(dlin):
        if row1 < dlin-1:
            for row2 in range(row1,dlin):
                if row2 < dlin-1:
                    for row3 in range(row2,dlin):
                        out_train[dlin + cnt + cte] = xb[row1,winstart:winstart + train_length] * xb[row2,winstart:winstart + train_length] * xb[row3,winstart:winstart + train_length]
                        cnt += 1

                else:
                    continue

        else:
            continue
        
    # the colunm of out_train_t is (x, b, x^2, bx, x^3, bx^2).T
    out_train_t = out_train[:,-1]

    x_train = out_train[:,:].T
    y_train = (xb[0:d, winstart + 1:winstart + train_length + 1] - xb[0:d, winstart:winstart + train_length]).T/dt

    model.set_weights(initial_weights)

    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.2, random_state=rand_seed)

    model.fit(x_train_split, y_train_split, validation_data=(x_val_split, y_val_split), epochs=35, batch_size=16)

    x_input = out_train_t.reshape(1,-1)
    
    delta_x = model.predict(x_input)[0][0]

    next_x = xb[0:d,winstart + train_length - 1] + delta_x*dt

    predict_traj_x.append(next_x[0])

    # estimating recovery rate
    x_t = tf.Variable(x_input, dtype=tf.float32)
    with tf.GradientTape() as tape:
        predictions = model(x_t)

    gradient_coef = tape.gradient(predictions, x_t).numpy()[0]

    W_out_matrix[i,:] = gradient_coef

    x_t = out_train_t[0]
    b_t = out_train_t[1]

    rrate = diff_f(W_out_matrix[i,:], x_t, b_t)

    estimate_rrate.append(rrate)

    i += 1

dic_pred = {'Time':tseries[train_length - 1 : -1],'b_param':b_tseries[train_length - 1 : -1],'next_x':x_tseries[train_length:],'pred_x':predict_traj_x,
            'rrate':rrate_tseries[train_length - 1 : -1],'esti_rrate':estimate_rrate}

polynomial = ['x', 'b', 'x^2', 'bx', 'x^3', 'bx^2']
for j in range(dtot + cte):
    dic_pred[polynomial[j]] = W_out_matrix[:,j]

csv_out = pd.DataFrame(dic_pred)
csv_out.to_csv('../results/fold/FNN_nonlin_fold.csv',header = True)
