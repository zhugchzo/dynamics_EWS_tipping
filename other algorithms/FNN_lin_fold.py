import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

if not os.path.exists('../results/fold'):
    os.makedirs('../results/fold')

length = 1000

train_length = 500 # Length of the rolling window
test_length = length - train_length

df_tseries = pd.read_csv('../data/fold/fold_train.csv')
keep_col_tseries = ['traj_x','b','Time','rrate','t_diff']
new_f_tseries = df_tseries[keep_col_tseries]
values_tseries = new_f_tseries.values

xb_tseries = np.array(values_tseries[:,:2])
tseries = np.array(values_tseries[:,2])
rrate_tseries = np.array(values_tseries[:,3])

dt = values_tseries[1,-1]

rand_seed = 25

np.random.seed(rand_seed)
tf.random.set_seed(rand_seed)
random.seed(rand_seed)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,), kernel_initializer='lecun_uniform'),
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer='lecun_uniform'),
        tf.keras.layers.Dense(1, kernel_initializer='lecun_uniform')
    ])
    return model

model = create_model()

optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer_adam, loss='mean_squared_error')

initial_weights = model.get_weights()

predict_traj_x = list()
estimate_rrate = list()

i = 0 # count

for winstart in range(test_length):

    x_train = xb_tseries[winstart : winstart + train_length]
    y_train = (xb_tseries[winstart+1 : winstart + train_length+1][:,0] - xb_tseries[winstart : winstart + train_length][:,0])/dt

    model.set_weights(initial_weights)

    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, test_size=0.2, random_state=rand_seed)

    model.fit(x_train_split, y_train_split, validation_data=(x_val_split, y_val_split), epochs=35, batch_size=16)

    x_input = x_train[-1].reshape(1,-1)
    
    delta_x = model.predict(x_input)[0][0]

    next_x = x_train[-1][0] + delta_x*dt

    predict_traj_x.append(next_x)

    # estimating recovery rate
    x_t = tf.Variable(x_input, dtype=tf.float32)
    with tf.GradientTape() as tape:
        predictions = model(x_t)

    rrate = tape.gradient(predictions, x_t).numpy()[0][0]

    estimate_rrate.append(rrate)

    i += 1

    print(i)

dic_pred = {'Time':tseries[train_length - 1 : -1],'b_param':xb_tseries[train_length - 1 : -1][:,1],'next_x':xb_tseries[train_length:][:,0],'pred_x':predict_traj_x,
            'rrate':rrate_tseries[train_length - 1 : -1],'esti_rrate':estimate_rrate}

csv_out = pd.DataFrame(dic_pred)
csv_out.to_csv('../results/fold/FNN_lin_fold.csv',header = True)
