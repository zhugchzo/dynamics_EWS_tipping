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

def de_fun(x, b, A, N, P):

    return P[0] + P[1]*x + P[2]*b + P[3]*x**2 + P[4]*b*x + P[5]*x**3 + P[6]*b*x**2 + np.sum(A*(P[7]*x + P[8]*b*x + P[9]*x.reshape(N,1)*x), axis=1)

def recov_fun(x, b, A, N, P):
    # Initialize the Jacobian matrix
    J = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                term_local = (P[1] + 2*P[3]*x[i] + P[4]*b + 3*P[5]*x[i]**2 + 2*P[6]*b*x[i])

                neighbor_sum = 0
                for k in range(N):
                    neighbor_sum += A[i, k] * x[k]
                
                term_coupling = P[9] * neighbor_sum
                
                J[i, j] = term_local + term_coupling

            else:
                
                J[i, j] = A[i, j] * (P[7] + P[8]*b + P[9]*x[i])

    try:
        evals = np.linalg.eigvals(J)
    except:
        return 1997.0209, 1997.0209, 1997.0209, 1997.0209 # if error; This particular number has no other significance, it's just the author's birthday

    evals = [lam for lam in evals]
    dom_eval = max(evals, key=lambda x: x.real)

    evals.remove(dom_eval)
    oth_eval = evals[0]

    rrate = dom_eval.real
    imag_dom_eval = dom_eval.imag

    real_oth_eval = oth_eval.real
    imag_oth_eval = oth_eval.imag
    
    return rrate, imag_dom_eval, real_oth_eval, imag_oth_eval

gap = 5

for network_size in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:

    if not os.path.exists('../results/fold_BA/bif/{}'.format(network_size)):
        os.makedirs('../results/fold_BA/bif/{}'.format(network_size))

    for random_seed in range(20):

        df_data = pd.read_csv('../results/ROC/fold_BA/ews/{}/dynamics_fold_BA_{}.csv'.format(network_size,random_seed))
        df_network = pd.read_csv('../data/fold_BA/ROC/{}/network_{}.csv'.format(network_size,random_seed), header=None)

        x0_sum = df_data['traj_x'].to_numpy()
        bif_pars_sum = df_data['b_param'].to_numpy()
        P_sum = df_data[['cte', 'x', 'b', 'x^2', 'bx', 'x^3', 'bx^2', 'x_j', 'bx_j', 'x_ix_j']].to_numpy()

        half_bif_pars = bif_pars_sum[0] - 0
        bif_value = bif_pars_sum[-1]
        scale_value = half_bif_pars*2

        initial_theta = df_data['initial_theta'].to_numpy()
        delta_theta = df_data['delta_theta'].to_numpy()

        thetal_sum = initial_theta + 499*delta_theta
        thetah_sum = initial_theta + 2000*delta_theta

        N = len(df_network)
        A = df_network.values

        sim_len = 100

        Time = [500+i for i in range(0, len(x0_sum), gap)]
        bif_type = []
        bif_error = []
        bif_type_value = []

        for ii in range(0, len(x0_sum), gap):

            t = np.arange(t0,sim_len,dt)

            x = np.zeros((N,len(t)))

            rrate = np.zeros(len(t))
            imag_dom_eval = np.zeros(len(t))
            real_oth_eval = np.zeros(len(t))
            imag_oth_eval = np.zeros(len(t))

            thetal = thetal_sum[ii]
            thetah = thetah_sum[ii]

            theta = pd.Series(np.linspace(thetal,thetah,len(t)),index=t)

            x0 = np.ones(N)*x0_sum[ii]
            P = P_sum[ii]
            bif_pars = bif_pars_sum[ii]

            # Run burn-in period on x0
            for j in range(int(tburn/dt)):

                x0 = x0 + de_fun(x0, theta[0], A, N, P)*dt

            rrate0, imag_dom_eval0, real_oth_eval0, imag_oth_eval0 = recov_fun(x0, theta[0], A, N, P)

            if rrate0 == 1997.0209 or np.any(np.isnan(x0)):
                print('Nan appears')

                # bif_error.append('NA')
                # bif_type.append('Blow')
                # bif_type_value.append(0)

                # break

            # Initial condition post burn-in period
            x[:,0] = x0

            rrate[0] = rrate0
            imag_dom_eval[0] = imag_dom_eval0
            real_oth_eval[0] = real_oth_eval0
            imag_oth_eval[0] = imag_oth_eval0

            rrate_record = []

            # Run simulation
            for i in range(len(t)-1):

                x[:,i+1] = x[:,i] + de_fun(x[:,i], theta.iloc[i], A, N, P)*dt

                rrate[i+1], imag_dom_eval[i+1], real_oth_eval[i+1], imag_oth_eval[i+1] = recov_fun(x[:,i+1], theta.iloc[i+1], A, N, P)

                if rrate[i+1] == 1997.0209:
                    print('rrate Nan appears')          
                    break

                if np.any(np.isnan(x[:,i+1])):
                    print('sim Nan appears')
                    break

                # Determine the tipping point by the recovery rate changing from negative to positive
                if rrate[i] < 0 and rrate[i+1] > 0:
                    rrate_record.append(i+1)
                    break

            if len(rrate_record) != 0:

                # print('{}:Bifurcation!'.format(ii))

                trans_time = rrate_record[0] # the time of tipping point
                type = imag_dom_eval[trans_time]

                theta_value = theta.iloc[trans_time]

                value = ((theta_value-thetal)*half_bif_pars)/(thetal-initial_theta[ii]) + bif_pars

                error = (value - bif_value) / scale_value

                bif_error.append(error)

                if type == 0:

                    bif_type.append('Steady')
                    bif_type_value.append(1)

                else:

                    bif_type.append('Hopf')
                    bif_type_value.append(0)

            else:
            
                # print('{}:No bifurcation!'.format(ii))

                bif_error.append('NA')
                bif_type.append('NA')
                bif_type_value.append(0)

        dic_bif = {'Time':Time, 'bif_type':bif_type, 'bif_error':bif_error, 'bif_type_value':bif_type_value}
        csv_out = pd.DataFrame(dic_bif)
        csv_out.to_csv('../results/fold_BA/bif/{}/bif_out_{}.csv'.format(network_size,random_seed),header = True)