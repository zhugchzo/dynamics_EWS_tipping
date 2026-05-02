import numpy as np
import pandas as pd
import os

# Simulation parameters
dt = 0.01
t0 = 0
tburn = 500 # burn-in period

#----------------------------------
# Simulate model
#----------------------------------

# Model

def de_fun(x, b, P):

    return P[0] + P[1]*x + P[2]*b + P[3]*x**2 + P[4]*b*x + P[5]*x**3 + P[6]*b*x**2

def recov_fun(x, b, P):

    return P[1] + 2*P[3]*x + P[4]*b + 3*P[5]*x**2 + 2*P[6]*b*x

if not os.path.exists('../results/analyze_bif/fold/bif'):
    os.makedirs('../results/analyze_bif/fold/bif')

gap = 5

for random_seed in range(200):

    df_data = pd.read_csv('../results/ROC/fold/ews/dynamics_fold_{}.csv'.format(random_seed))

    x0_sum = df_data['traj_x'].to_numpy()
    bif_pars_sum = df_data['b'].to_numpy()
    P_sum = df_data[['cte', 'x', 'b', 'x^2', 'bx', 'x^3', 'bx^2']].to_numpy()

    half_bif_pars = bif_pars_sum[0] - 1
    bif_value = bif_pars_sum[-1]
    scale_value = half_bif_pars*2

    initial_theta = df_data['initial_theta'].to_numpy()
    delta_theta = df_data['delta_theta'].to_numpy()

    thetal_sum = initial_theta + 499*delta_theta
    thetah_sum = initial_theta + 2000*delta_theta

    sim_len = 100

    Time = [500+i for i in range(0, len(x0_sum), gap)]
    bif_type = []
    bif_error = []
    bif_type_value = []

    for ii in range(0, len(x0_sum), gap):

        t = np.arange(t0,sim_len,dt)

        x = np.zeros(len(t))

        rrate = np.zeros(len(t))

        thetal = thetal_sum[ii]
        thetah = thetah_sum[ii]

        theta = pd.Series(np.linspace(thetal,thetah,len(t)),index=t)

        x0 = x0_sum[ii]
        P = P_sum[ii]
        bif_pars = bif_pars_sum[ii]

        # Run burn-in period on x0
        for j in range(int(tburn/dt)):

            x0 = x0 + de_fun(x0, theta[0], P)*dt

        rrate0 = recov_fun(x0, theta[0], P)

        if rrate0 == 1997.0209 or np.any(np.isnan(x0)):
            print('Nan appears')

            # bif_error.append('NA')
            # bif_type.append('Blow')
            # bif_type_value.append(0)

            # break

        # Initial condition post burn-in period
        x[0] = x0

        rrate[0] = rrate0

        rrate_record = []

        # Run simulation
        for i in range(len(t)-1):

            x[i+1] = x[i] + de_fun(x[i], theta.iloc[i], P)*dt

            rrate[i+1] = recov_fun(x[i+1], theta.iloc[i+1], P)

            if rrate[i+1] == 1997.0209:
                print('rrate Nan appears')
                break

            if np.any(np.isnan(x[i+1])):
                print('sim Nan appears')
                break

            # Determine the tipping point by the recovery rate changing from negative to positive
            if rrate[i] < 0 and rrate[i+1] > 0:
                rrate_record.append(i+1)
                break

        if len(rrate_record) != 0:

            # print('{}:Bifurcation!'.format(ii))

            fold_time = rrate_record[0] # the time of tipping point

            theta_value = theta.iloc[fold_time]

            value = ((theta_value-thetal)*half_bif_pars)/(thetal-initial_theta[ii]) + bif_pars

            error = (value - bif_value) / scale_value

            bif_error.append(error)

            bif_type.append('Steady')
            bif_type_value.append(1)
            
        else:
        
            # print('{}:No bifurcation!'.format(ii))

            bif_error.append('NA')
            bif_type.append('NA')
            bif_type_value.append(0)

    dic_bif = {'Time':Time, 'bif_type':bif_type, 'bif_error':bif_error, 'bif_type_value':bif_type_value}
    csv_out = pd.DataFrame(dic_bif)
    csv_out.to_csv('../results/analyze_bif/fold/bif/bif_out_{}.csv'.format(random_seed),header = True)