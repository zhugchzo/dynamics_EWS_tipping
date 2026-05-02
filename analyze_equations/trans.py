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

def de_fun(x, y, b, P):

    return (P[0]+P[1]*x+P[2]*y+P[3]*b+P[4]*x**2+P[5]*x*y+P[6]*b*x+P[7]*y**2+P[8]*b*y+P[9]*x**3+
            P[10]*x**2*y+P[11]*b*x**2+P[12]*x*y**2+P[13]*b*x*y+P[14]*y**3+P[15]*b*y**2)

def recov_fun(x, y, b, Px, Py):

    j11 = (Px[1] + 2*Px[4]*x + Px[5]*y + Px[6]*b + 3*Px[9]*x**2
    + 2*Px[10]*x*y + 2*Px[11]*b*x + Px[12]*y**2 + Px[13]*b*y)

    j12 = (Px[2] + Px[5]*x + 2*Px[7]*y + Px[8]*b + Px[10]*x**2
    + 2*Px[12]*x*y + Px[13]*b*x + 3*Px[14]*y**2 + 2*Px[15]*b*y)

    j21 = (Py[1] + 2*Py[4]*x + Py[5]*y + Py[6]*b + 3*Py[9]*x**2
    + 2*Py[10]*x*y + 2*Py[11]*b*x + Py[12]*y**2 + Py[13]*b*y)

    j22 = (Py[2] + Py[5]*x + 2*Py[7]*y + Py[8]*b + Py[10]*x**2
    + 2*Py[12]*x*y + Py[13]*b*x + 3*Py[14]*y**2 + 2*Py[15]*b*y)

    J = np.array([[j11,j12],[j21,j22]])

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

if not os.path.exists('../results/analyze_bif/trans/bif'):
    os.makedirs('../results/analyze_bif/trans/bif')

gap = 5

for random_seed in range(200):

    df_data = pd.read_csv('../results/ROC/trans/ews/dynamics_trans_{}.csv'.format(random_seed))

    x0_sum = df_data['traj_x'].to_numpy()
    y0_sum = df_data['traj_y'].to_numpy()
    bif_pars_sum = df_data['b'].to_numpy()
    Px_sum = df_data[['cte_f1', 'x_f1', 'y_f1', 'b_f1', 'x^2_f1', 'xy_f1', 'bx_f1', 'y^2_f1', 'by_f1', 'x^3_f1', 'x^2y_f1', 'bx^2_f1', 'xy^2_f1', 'bxy_f1', 'y^3_f1', 'by^2_f1']].to_numpy()
    Py_sum = df_data[['cte_f2', 'x_f2', 'y_f2', 'b_f2', 'x^2_f2', 'xy_f2', 'bx_f2', 'y^2_f2', 'by_f2', 'x^3_f2', 'x^2y_f2', 'bx^2_f2', 'xy^2_f2', 'bxy_f2', 'y^3_f2', 'by^2_f2']].to_numpy()

    half_bif_pars = bif_pars_sum[0] - 0
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
        y = np.zeros(len(t))

        rrate = np.zeros(len(t))
        imag_dom_eval = np.zeros(len(t))
        real_oth_eval = np.zeros(len(t))
        imag_oth_eval = np.zeros(len(t))

        thetal = thetal_sum[ii]
        thetah = thetah_sum[ii]

        theta = pd.Series(np.linspace(thetal,thetah,len(t)),index=t)

        x0 = x0_sum[ii]
        y0 = y0_sum[ii]
        Px = Px_sum[ii]
        Py = Py_sum[ii]
        bif_pars = bif_pars_sum[ii]

        # Run burn-in period on x0
        for j in range(int(tburn/dt)):

            x0 = x0 + de_fun(x0, y0, theta[0], Px)*dt
            y0 = y0 + de_fun(x0, y0, theta[0], Py)*dt

        rrate0, imag_dom_eval0, real_oth_eval0, imag_oth_eval0 = recov_fun(x0, y0, theta[0], Px, Py)

        if rrate0 == 1997.0209 or np.any(np.isnan(x0)):
            print('Nan appears')

            # bif_error.append('NA')
            # bif_type.append('Blow')
            # bif_type_value.append(0)

            # break

        # Initial condition post burn-in period
        x[0] = x0
        y[0] = x0

        rrate[0] = rrate0
        imag_dom_eval[0] = imag_dom_eval0
        real_oth_eval[0] = real_oth_eval0
        imag_oth_eval[0] = imag_oth_eval0

        rrate_record = []

        # Run simulation
        for i in range(len(t)-1):

            x[i+1] = x[i] + de_fun(x[i], y[i], theta.iloc[i], Px)*dt
            y[i+1] = y[i] + de_fun(x[i], y[i], theta.iloc[i], Py)*dt

            rrate[i+1], imag_dom_eval[i+1], real_oth_eval[i+1], imag_oth_eval[i+1] = recov_fun(x[i+1], y[i+1], theta.iloc[i+1], Px, Py)

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
    csv_out.to_csv('../results/analyze_bif/trans/bif/bif_out_{}.csv'.format(random_seed),header = True)