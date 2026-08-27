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

fig, axs = plt.subplots(2, 1, figsize=(11,15))

(ax1, ax2) = axs.flatten()

df = pandas.read_csv('../data/fold/fold.csv')
df_sindy_results = pandas.read_csv('../results/mean_lambda/fold_SINDy_mean_lambda.csv')

truncate_length = 1185

t1 = df['Time'][:truncate_length] + 1
t2 = df_sindy_results['Time'] + 501

x = df['traj_x'][:truncate_length]
sindy_esti_rrate = df_sindy_results['esti_mean']
sindy_esti_re_lower = df_sindy_results['esti_ci_lower']
sindy_esti_re_upper = df_sindy_results['esti_ci_upper']

train_length = 500
length = 1000

ax1.scatter(t1[::5],x[::5],s=40,c='darkslateblue',marker='o')
ax1.plot(t1[::5],x[::5],c='darkslateblue',linewidth=3)
ax1.scatter(t1[train_length:length][::5],x[train_length:length][::5],s=40,c='darkgrey',marker='o',zorder=4)
ax1.plot(t1[train_length:length][::5],x[train_length:length][::5],c='darkgrey',linewidth=3)
ax1.scatter(1000,x[1000],s=500, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax1.set_ylabel('System state',font_y)
ax1.yaxis.set_label_coords(-0.15, 0.5)
ax1.set_xlim(-50,1220)
ax1.set_ylim(-1,9)
ax1.set_yticks([0,4,8])
ax1.set_yticklabels(['0','4','8'])
ax1.tick_params(axis='x', bottom=False, labelbottom=False)
ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)
# ax1.text(850,3.7,'fold',size=40)
ax1.text(0.85,0.7,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
ax1.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)

ax1.plot([15, 484], [6, 6], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 6), (494, 6), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=30)
ax1.add_patch(arrow1)

ax1.text(240,5.3,r'$L$', fontsize=40, ha='center', va='center', color='black')

# ax2.scatter(t2,sindy_esti_rrate,s=80,color='crimson',marker='o',edgecolor=None,alpha=0.3,zorder=5)
# ax2.scatter(t2,sindy_esti_rrate,s=80,marker='o',facecolors='none',edgecolors='crimson',zorder=5)
ax2.plot(t2,sindy_esti_rrate,c='crimson',linewidth=6,alpha=0.9,zorder=2)
ax2.fill_between(t2,sindy_esti_re_lower,sindy_esti_re_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
ax2.set_xlabel('Time',font_x,labelpad=15)
ax2.set_ylabel('Re',font_y)
ax2.yaxis.set_label_coords(-0.15, 0.5)
ax2.set_xlim(-50,1220)
ax2.set_xticks([0,500,1000])
ax2.set_xticklabels(['0','500','1000'])
ax2.set_ylim(-0.48,0.048)
ax2.set_yticks([-0.4,-0.2,0])
ax2.set_yticklabels(['-0.4','-0.2','0'])
ax2.tick_params(axis='x', pad=10)
ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)
ax2.text(0.85,0.7,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
ax2.axvline(x=500, color='black', linestyle='--',alpha=0.5, linewidth=2)
ax2.hlines(y=0, xmin=500, xmax=1150, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

legend_state = mlines.Line2D([], [], color='darkslateblue', linestyle='-',marker='o', linewidth=3, markersize=15)
legend_pstate = mlines.Line2D([], [], color='darkgrey', linestyle='-',marker='o', linewidth=3, markersize=15)
legend_bif = mlines.Line2D([], [], linestyle='None', marker='o',markerfacecolor='white', markeredgecolor='black',
    markersize=20, markeredgewidth=3)
legend_esti_re = mlines.Line2D([], [], color='crimson', linestyle='-',linewidth=6,alpha=0.9)

ax1.legend(
    handles=[legend_state, legend_pstate, legend_bif],
    labels=['State','Forecast regime','Fold bifurcation'],
    loc='lower left', bbox_to_anchor=(0.05, 0.1), frameon=False, 
    prop=font_manager.FontProperties(family='Arial', size=36, weight='normal')
)

ax2.legend(
    handles=[legend_esti_re],
    labels=['Estimated Re'],
    loc='lower left', bbox_to_anchor=(0.05, 0.3), frameon=False, 
    prop=font_manager.FontProperties(family='Arial', size=36, weight='normal')
)

plt.subplots_adjust(top=0.99, bottom=0.1, left=0.18, right=0.98, hspace=0.1)
plt.savefig('../figures/SFIG2.pdf',format='pdf')