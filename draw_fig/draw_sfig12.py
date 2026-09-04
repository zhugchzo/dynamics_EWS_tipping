import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y = {'family':'Arial','weight':'normal','size': 40}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36

fig, axs = plt.subplots(3, 1, figsize=(12,22))

ax1, ax2, ax3 = axs[0], axs[1], axs[2]

df_hopf = pandas.read_csv('../data/3D_hopf/3D_hopf.csv')
df_hopf_results0 = pandas.read_csv('../results/mean_lambda/partial/3D_hopf_mean_lambda_0.csv')
df_hopf_results1 = pandas.read_csv('../results/mean_lambda/partial/3D_hopf_mean_lambda_1.csv')
df_hopf_results2 = pandas.read_csv('../results/mean_lambda/partial/3D_hopf_mean_lambda_2.csv')

truncate_length = 1185

t1 = df_hopf['Time'][:truncate_length] + 1
t2 = df_hopf_results0['Time'] + 501

hopf_x = df_hopf['traj_x'][:truncate_length]

hopf_esti_rrate0 = df_hopf_results0['esti_re_mean']
hopf_esti_re_lower0 = df_hopf_results0['esti_re_ci_lower']
hopf_esti_re_upper0 = df_hopf_results0['esti_re_ci_upper']
hopf_esti_im10 = df_hopf_results0['esti_im1_mean']
hopf_esti_im1_lower0 = df_hopf_results0['esti_im1_ci_lower']
hopf_esti_im1_upper0 = df_hopf_results0['esti_im1_ci_upper']
hopf_esti_im20 = df_hopf_results0['esti_im2_mean']
hopf_esti_im2_lower0 = df_hopf_results0['esti_im2_ci_lower']
hopf_esti_im2_upper0 = df_hopf_results0['esti_im2_ci_upper']

hopf_esti_rrate1 = df_hopf_results1['esti_re_mean']
hopf_esti_re_lower1 = df_hopf_results1['esti_re_ci_lower']
hopf_esti_re_upper1 = df_hopf_results1['esti_re_ci_upper']
hopf_esti_im11 = df_hopf_results1['esti_im1_mean']
hopf_esti_im1_lower1 = df_hopf_results1['esti_im1_ci_lower']
hopf_esti_im1_upper1 = df_hopf_results1['esti_im1_ci_upper']
hopf_esti_im21 = df_hopf_results1['esti_im2_mean']
hopf_esti_im2_lower1 = df_hopf_results1['esti_im2_ci_lower']
hopf_esti_im2_upper1 = df_hopf_results1['esti_im2_ci_upper']

hopf_esti_rrate2 = df_hopf_results2['esti_re_mean']
hopf_esti_re_lower2 = df_hopf_results2['esti_re_ci_lower']
hopf_esti_re_upper2 = df_hopf_results2['esti_re_ci_upper']
hopf_esti_im12 = df_hopf_results2['esti_im1_mean']
hopf_esti_im1_lower2 = df_hopf_results2['esti_im1_ci_lower']
hopf_esti_im1_upper2 = df_hopf_results2['esti_im1_ci_upper']
hopf_esti_im22 = df_hopf_results2['esti_im2_mean']
hopf_esti_im2_lower2 = df_hopf_results2['esti_im2_ci_lower']
hopf_esti_im2_upper2 = df_hopf_results2['esti_im2_ci_upper']

train_length = 500
length = 1000

ax1.scatter(t1[::5],hopf_x[::5],s=40,c='darkslateblue',marker='o')
ax1.plot(t1[::5],hopf_x[::5],c='darkslateblue',linewidth=3)
ax1.scatter(t1[train_length:length][::5],hopf_x[train_length:length][::5],s=40,c='darkgrey',marker='o',zorder=4)
ax1.plot(t1[train_length:length][::5],hopf_x[train_length:length][::5],c='darkgrey',linewidth=3)
ax1.scatter(1000,hopf_x[1000],s=500, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax1.set_ylabel('System state',font_y)
ax1.yaxis.set_label_coords(-0.11, 0.5)
ax1.set_xlim(-50,1220)
ax1.set_ylim(-0.15,0.15)
ax1.set_yticks([-0.1,0,0.1])
ax1.set_yticklabels(['-0.1','0','0.1'])
ax1.tick_params(axis='x', bottom=False, labelbottom=False)
ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)

ax1.text(0.85,0.75,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax1.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)

ax1.plot([15, 484], [0.12, 0.12], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 0.12), (494, 0.12), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=30)
ax1.add_patch(arrow1)

ax1.text(240,0.1,r'$L$', fontsize=40, ha='center', va='center', color='black')

ax2.plot(t2,hopf_esti_rrate0,c='crimson',linewidth=6,alpha=0.9,zorder=3)
ax2.fill_between(t2,hopf_esti_re_lower0,hopf_esti_re_upper0,color='crimson',alpha=0.3,linewidth=0,zorder=3)

ax2.plot(t2,hopf_esti_rrate1,c='darkviolet',linewidth=6,alpha=0.9,zorder=2)
ax2.fill_between(t2,hopf_esti_re_lower1,hopf_esti_re_upper1,color='darkviolet',alpha=0.3,linewidth=0,zorder=2)

ax2.plot(t2,hopf_esti_rrate2,c='darkorange',linewidth=6,alpha=0.9,zorder=1)
ax2.fill_between(t2,hopf_esti_re_lower2,hopf_esti_re_upper2,color='darkorange',alpha=0.3,linewidth=0,zorder=1)
ax2.set_ylabel('Re',font_y)
ax2.yaxis.set_label_coords(-0.11, 0.5)
ax2.set_xlim(-50,1220)
ax2.set_ylim(-0.48,0.048)
ax2.set_yticks([-0.4,-0.2,0])
ax2.set_yticklabels(['-0.4','-0.2','0'])
ax2.tick_params(axis='x', bottom=False, labelbottom=False)
ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)

ax2.text(0.85,0.75,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax2.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax2.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

ax3.plot(t2,hopf_esti_im10,c='crimson',linewidth=6,alpha=0.9,zorder=3)
ax3.plot(t2,hopf_esti_im20,c='crimson',linewidth=6,alpha=0.9,zorder=3)
ax3.fill_between(t2,hopf_esti_im1_lower0,hopf_esti_im1_upper0,color='crimson',alpha=0.3,linewidth=0,zorder=3)
ax3.fill_between(t2,hopf_esti_im2_lower0,hopf_esti_im2_upper0,color='crimson',alpha=0.3,linewidth=0,zorder=3)

ax3.plot(t2,hopf_esti_im11,c='darkviolet',linewidth=6,alpha=0.9,zorder=2)
ax3.plot(t2,hopf_esti_im21,c='darkviolet',linewidth=6,alpha=0.9,zorder=2)
ax3.fill_between(t2,hopf_esti_im1_lower1,hopf_esti_im1_upper1,color='darkviolet',alpha=0.3,linewidth=0,zorder=2)
ax3.fill_between(t2,hopf_esti_im2_lower1,hopf_esti_im2_upper1,color='darkviolet',alpha=0.3,linewidth=0,zorder=2)

ax3.plot(t2,hopf_esti_im12,c='darkorange',linewidth=6,alpha=0.9,zorder=1)
ax3.plot(t2,hopf_esti_im22,c='darkorange',linewidth=6,alpha=0.9,zorder=1)
ax3.fill_between(t2,hopf_esti_im1_lower2,hopf_esti_im1_upper2,color='darkorange',alpha=0.3,linewidth=0,zorder=1)
ax3.fill_between(t2,hopf_esti_im2_lower2,hopf_esti_im2_upper2,color='darkorange',alpha=0.3,linewidth=0,zorder=1)

ax3.set_xlabel('Time',font_x,labelpad=15)
ax3.set_ylabel('Im',font_y)
ax3.yaxis.set_label_coords(-0.11, 0.5)
ax3.set_xlim(-50,1220)
ax3.set_xticks([0,500,1000])
ax3.set_xticklabels(['0','500','1000'])
ax3.set_ylim(-1,1)
ax3.set_yticks([-0.5,0,0.5])
ax3.set_yticklabels(['-0.5','0','0.5'])
ax3.tick_params(axis='x', pad=10)
ax3.tick_params(direction='in', length=10)
ax3.minorticks_on()
ax3.tick_params(axis='x', which='minor', bottom=False)
ax3.tick_params(axis='y', which='minor', direction='in', length=5)
ax3.text(0.85,0.75,'(c)',ha='left', transform=ax3.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax3.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax3.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

legend_state = mlines.Line2D([], [], color='darkslateblue', linestyle='-',marker='o', linewidth=3, markersize=15)
legend_pstate = mlines.Line2D([], [], color='darkgrey', linestyle='-',marker='o', linewidth=3, markersize=15)
legend_bif = mlines.Line2D([], [], linestyle='None', marker='o',markerfacecolor='white', markeredgecolor='black',
    markersize=20, markeredgewidth=3)

legend_esti_re0 = mlines.Line2D([], [], color='crimson', linestyle='-',linewidth=6,alpha=0.9)
legend_esti_re1 = mlines.Line2D([], [], color='darkviolet', linestyle='-',linewidth=6,alpha=0.9)
legend_esti_re2 = mlines.Line2D([], [], color='darkorange', linestyle='-',linewidth=6,alpha=0.9)

legend_esti_im0 = mlines.Line2D([], [], color='crimson', linestyle='-',linewidth=6,alpha=0.9)
legend_esti_im1 = mlines.Line2D([], [], color='darkviolet', linestyle='-',linewidth=6,alpha=0.9)
legend_esti_im2 = mlines.Line2D([], [], color='darkorange', linestyle='-',linewidth=6,alpha=0.9)

ax1.legend(
    handles=[legend_state, legend_pstate, legend_bif],
    labels=['State','Forecast regime','Hopf bifurcation'],
    loc='lower left', bbox_to_anchor=(0.3, 0.03), frameon=False, 
    prop=font_manager.FontProperties(family='Arial', size=30, weight='normal')
)

ax2.legend(
    handles=[legend_esti_re0,  legend_esti_re1,  legend_esti_re2],
    labels=['Re estimated\nby $(I,C)$', 'Re estimated\nby $(I,N)$', 'Re estimated\nby $(C,N)$'],
    loc='lower left', bbox_to_anchor=(0.01, 0.2), frameon=False, 
    prop=font_manager.FontProperties(family='Arial', size=30, weight='normal')
)

ax3.legend(
    handles=[legend_esti_im0,  legend_esti_im1,  legend_esti_im2],
    labels=['Im estimated\nby $(I,C)$', 'Im estimated\nby $(I,N)$', 'Im estimated\nby $(C,N)$'],
    loc='lower left', bbox_to_anchor=(0.01, 0.2), frameon=False, 
    prop=font_manager.FontProperties(family='Arial', size=30, weight='normal')
)

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.15, right=0.98, hspace=0.15)
plt.savefig('../figures/SFIG12.pdf',format='pdf')