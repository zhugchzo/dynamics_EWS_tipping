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
df_transient_results = pandas.read_csv('../results/mean_lambda/transient/fold_mean_lambda.csv')

transient_begin = 995
transient_over = 1032

t1 = df['Time'][transient_begin:transient_over] + 1
t2 = df_transient_results['Time'] + 1000
t_transient = list(t2 - 1)

x = df['traj_x'][transient_begin:transient_over]

transient_true_rrate = df_transient_results['true_re_mean']
transient_esti_rrate = df_transient_results['esti_re_mean']
transient_esti_re_lower = df_transient_results['esti_re_ci_lower']
transient_esti_re_upper = df_transient_results['esti_re_ci_upper']

ax1.scatter(t1,x,s=150,c='darkslateblue',marker='o')
ax1.plot(t1,x,c='darkslateblue',linewidth=6)
ax1.scatter(t2,x[t_transient],s=150,c='darkgrey',marker='o',zorder=4)
ax1.plot(t2,x[t_transient],c='darkgrey',linewidth=6)
ax1.scatter(1000,x[1000],s=1000, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax1.set_ylabel('System state',font_y)
ax1.yaxis.set_label_coords(-0.15, 0.5)
ax1.set_xlim(993,1034)
ax1.set_xticks([1000,1027])
ax1.set_xticklabels(['1000','1027'])
ax1.set_ylim(-0.2,4.8)
ax1.set_yticks([0,2,4])
ax1.set_yticklabels(['0','2','4'])
ax1.tick_params(axis='x', pad=10)
ax1.tick_params(direction='in', length=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(axis='y', which='minor', direction='in', length=5)
# ax1.text(850,3.7,'fold',size=40)
ax1.text(0.8,0.85,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})

# ax1.axvline(x=0, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
# ax1.axvline(x=500, color='black', linestyle='--', linewidth=2,alpha=0.5, zorder=0)
# ax1.plot([15, 484], [6, 6], linestyle='--', color='black', linewidth=2)
# arrow1 = FancyArrowPatch((5, 6), (494, 6), 
#                         arrowstyle='<|-|>', 
#                         color='black',
#                         linewidth = 0, 
#                         mutation_scale=30)
# ax1.add_patch(arrow1)

# ax1.text(240,5.3,r'$L$', fontsize=40, ha='center', va='center', color='black')

# ax2.scatter(t2,transient_esti_rrate,s=150,color='crimson',marker='o',edgecolor=None,alpha=0.3,zorder=5)
# ax2.scatter(t2,transient_esti_rrate,s=150,marker='o',facecolors='none',edgecolors='crimson',zorder=5)

ax2.plot(t2,transient_true_rrate,c='dimgrey',linewidth=6,alpha=0.9,zorder=1)
ax2.plot(t2,transient_esti_rrate,c='crimson',linewidth=6,alpha=0.9,zorder=2)
ax2.fill_between(t2,transient_esti_re_lower,transient_esti_re_upper,color='crimson',alpha=0.3,linewidth=0,zorder=2)
ax2.set_xlabel('Time',font_x,labelpad=5)
ax2.set_ylabel('Re',font_y)
ax2.yaxis.set_label_coords(-0.15, 0.5)
ax2.set_xlim(993,1034)
ax2.set_xticks([1000,1027])
ax2.set_xticklabels(['1000','1027'])
ax2.set_ylim(-0.7,0.25)
ax2.set_yticks([-0.6,0,0.2])
ax2.set_yticklabels(['-0.6','0','0.2'])
ax2.tick_params(axis='x', pad=10)
ax2.tick_params(direction='in', length=10)
ax2.minorticks_on()
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.tick_params(axis='y', which='minor', direction='in', length=5)
ax2.text(0.8,0.85,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':60,'weight':'normal'})
# ax2.axvline(x=500, color='black', linestyle='--',alpha=0.5, linewidth=2)
ax2.hlines(y=0, xmin=991, xmax=1037, color='grey', linestyle='--', linewidth=4, alpha=0.5, zorder=0)

legend_state = mlines.Line2D([], [], color='darkslateblue', linestyle='-',marker='o', linewidth=3, markersize=15)
legend_pstate = mlines.Line2D([], [], color='darkgrey', linestyle='-',marker='o', linewidth=3, markersize=15)
legend_bif = mlines.Line2D([], [], linestyle='None', marker='o',markerfacecolor='white', markeredgecolor='black',
    markersize=20, markeredgewidth=3)
legend_true_re = mlines.Line2D([], [], color='dimgrey', linestyle='-',linewidth=6,alpha=0.9)
legend_esti_re = mlines.Line2D([], [], color='crimson', linestyle='-',linewidth=6,alpha=0.9)

ax1.legend(
    handles=[legend_state, legend_pstate, legend_bif],
    labels=['State','Forecast regime\n(transient regime)','Fold bifurcation'],
    loc='lower left', bbox_to_anchor=(0.05, 0.1), frameon=False, 
    prop=font_manager.FontProperties(family='Arial', size=36, weight='normal')
)

ax2.legend(
    handles=[legend_true_re, legend_esti_re],
    labels=['True Re', 'Estimated Re'],
    loc='lower left', bbox_to_anchor=(0.05, 0.1), frameon=False, 
    prop=font_manager.FontProperties(family='Arial', size=36, weight='normal')
)

plt.subplots_adjust(top=0.99, bottom=0.08, left=0.18, right=0.98, hspace=0.2)
plt.savefig('../figures/SFIG3.pdf',format='pdf')