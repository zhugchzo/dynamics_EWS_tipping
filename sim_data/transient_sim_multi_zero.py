# import python libraries
import numpy as np
import pandas as pd
import os

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period
sigma = 0.01 # noise intensity

#----------------------------------
# Simulate model
#----------------------------------

# Model

def q_fun(x, model):
    if model == 'rational':
        w = 0.06
        u = (x + 0.4)/w
        denom = 4 + u**2
        q = 2*(1 + u**2)/denom
        dq = 12*u/(w*denom**2)
        return q, dq

    if model != 'gaussian':
        raise ValueError(f'Unknown model: {model}')

    center, w, scale = -0.35, 0.08, 1
    z = x - center
    bump = np.exp(-(z/w)**2)
    q = scale*np.exp(-bump)
    dq = q*(2*z/w**2)*bump
    return q, dq



def de_fun(x, b, model):
    q, _ = q_fun(x, model)
    return q*(b + x - x**3)


def recov_fun(x, b, model):
    q, dq = q_fun(x, model)
    return q*(1 - 3*x**2) + dq*(b + x - x**3)


# Model parameters
bcrit = 2/(3*np.sqrt(3))
bl = 0.0
bh = 0.45 # Bifurcation parameter high.
c_rate = 1.5e-5 # Bifurcation parameter increment per step.

sim_len = int((bh-bl)*dt/c_rate)
series_len = 1000
base_dir = '../data/transient/multi_zero'

for model in ('gaussian', 'rational'):
    output_dir = os.path.join(base_dir, model)
    os.makedirs(output_dir, exist_ok=True)

    for random_seed in range(200):

        np.random.seed(random_seed)

        t = np.arange(t0,sim_len,dt)
        x0 = -1.0 # Low stable equilibrium at bl = 0.
        equi0 = x0

        x = np.zeros(len(t))
        equi = np.zeros(len(t))
        rrate = np.zeros(len(t))

        b = np.linspace(bl,bh,len(t))

        # Create brownian increments (s.d. sqrt(dt))
        dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
        dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))

        # Run burn-in period on x0
        for i in range(int(tburn/dt)):
            x0 = x0 + de_fun(x0,bl,model)*dt + dW_burn[i]
            equi0 = equi0 + de_fun(equi0,bl,model)*dt

        rrate0 = recov_fun(equi0,bl,model)

        # Initial condition post burn-in period
        x[0]=x0
        equi[0]=equi0
        rrate[0]=rrate0

        rrate1_record = []
        rrate2_record = []

        # Run simulation
        for i in range(len(t)-1):
            x[i+1] = x[i] + de_fun(x[i],b[i],model)*dt + dW[i]
            equi[i+1] = equi[i] + de_fun(equi[i],b[i],model)*dt
            rrate[i+1] = recov_fun(equi[i+1],b[i+1],model)
            # Detect crossings after the parameter passes the fold.
            if b[i] > bcrit:
                if rrate[i] <= 0 < rrate[i+1]:
                    rrate1_record.append(i+1)
                if rrate[i] >= 0 > rrate[i+1]:
                    rrate2_record.append(i+1)

        if rrate1_record and rrate2_record:

            trans_time = rrate1_record[0] # First negative-to-positive crossing.
            last_crossing = max(rrate1_record[-1], rrate2_record[-1])
            relaxed = np.flatnonzero(np.abs(np.diff(equi[last_crossing:])) < 1e-3)
            relax_time = last_crossing + 1 + relaxed[0] if relaxed.size else len(t)-1

            # 第一阶段：0 到 trans_time，共 series_len 个采样点
            c1 = np.linspace(0, trans_time, series_len)

            # 第一阶段的采样间隔
            delta_t = trans_time / (series_len - 1)

            # 第二阶段所需的采样间隔数
            relax_intervals = round((relax_time - trans_time) / delta_t)

            # 第二阶段：避免重复包含 trans_time
            c0 = np.linspace(
                trans_time,
                relax_time,
                relax_intervals + 1
            )[1:]

            # 拼接完整时间序列
            c2 = np.concatenate((c1, c0))

            c2 = np.rint(c2).astype(int).tolist()

        else:
            print(f'{model}, seed {random_seed}: missing post-fold zero crossing')
            continue

        # Store series data in a temporary DataFrame
        data = {'t':t,'traj_x': x,'equix':equi,'b':b,'rrate': rrate}
        df_temp = pd.DataFrame(data)

        df_temp_c = df_temp.iloc[c2].copy()

        df_temp_c['Time'] = range(len(df_temp_c))

        df_temp_c['t_diff'] = df_temp_c['t'].diff()

        df_temp_c['rrate_cubic'] = 1 - 3*df_temp_c['traj_x']**2

        cols = ['Time'] + [col for col in df_temp_c.columns if col != 'Time']

        df_temp_c = df_temp_c[cols].iloc[500:]

        print(f'{model}, seed {random_seed}: {len(rrate1_record)} negative-to-positive, '
              f'{len(rrate2_record)} positive-to-negative crossings')

        #------------------------------------
        # Export data 
        #-----------------------------------

        filepath = os.path.join(output_dir, 'multi_zero_{}.csv'.format(random_seed))

        df_temp_c.to_csv(filepath,index=False)
