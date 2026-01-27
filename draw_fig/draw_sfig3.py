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

fig, axs = plt.subplots(2, 1, figsize=(11,16))

(ax1, ax2) = axs.flatten()

df = pandas.read_csv('../data/fold/fold.csv')
df_sindy_results = pandas.read_csv('../results/fold/SINDy_fold.csv')

t1 = df['Time'] + 1
t2 = df_sindy_results['Time'] + 1

x = df['traj_x']

rrate = df_sindy_results['rrate']
sindy_esti_rrate = df_sindy_results['esti_rrate']

train_length = 500
length = 1000

ax1.scatter(t1[::5],x[::5],s=40,c='darkslateblue',marker='o')
ax1.plot(t1[::5],x[::5],c='darkslateblue',linewidth=3)
ax1.scatter(t1[train_length:length][::5],x[train_length:length][::5],s=40,c='crimson',marker='o')
ax1.plot(t1[train_length:length][::5],x[train_length:length][::5],c='crimson',linewidth=3)
ax1.scatter(1000,x[1000],s=500, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)
ax1.set_ylabel('System state',font_y,labelpad=45)
ax1.set_xlim(-55,1100)
ax1.set_xticks([0,1000])
ax1.set_ylim(-0.1,9)
ax1.set_yticks([1,8])
ax1.set_yticklabels(['1','8'])
ax1.tick_params(axis='x', bottom=False, labelbottom=False)
ax1.tick_params(direction='in', length=10)
ax1.text(850,3.7,'fold',size=40)
ax1.text(0.06,0.93,'(a)',ha='left', transform=ax1.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax1.axvline(x=500, color='black', linestyle='--', linewidth=2)

ax1.plot([15, 484], [6, 6], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 6), (494, 6), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=30)
ax1.add_patch(arrow1)

ax1.text(240,5.5,'sliding window', fontsize=32, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((450, 5.5), (490, 5.5), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=20)
ax1.add_patch(arrow2)

ax2.plot(t2,rrate,c='darkviolet',linewidth=8,alpha=0.8)
ax2.scatter(t2,sindy_esti_rrate,s=80,marker='o',facecolors='none',edgecolors='crimson',zorder=5)
ax2.set_xlabel('Timepoint',font_x)
ax2.set_ylabel('Re',font_y,labelpad=0)
ax2.set_xlim(-55,1100)
ax2.set_xticks([0,500,1000])
ax2.set_xticklabels(['0','500','1000'])
ax2.set_ylim(-0.55,0.05)
ax2.set_yticks([-0.5,0])
ax2.set_yticklabels(['-0.5','0'])
ax2.tick_params(direction='in', length=10)
ax2.tick_params(axis='x', pad=10)
ax2.text(0.06,0.93,'(b)',ha='left', transform=ax2.transAxes,fontdict={'family':'Arial','size':40,'weight':'normal'})
ax2.axvline(x=500, color='black', linestyle='--', linewidth=2)

legend_state = mlines.Line2D([], [], color='darkslateblue', linestyle='-',marker='o', linewidth=3, markersize=15)
legend_pstate = mlines.Line2D([], [], color='crimson', linestyle='-',marker='o', linewidth=3, markersize=15)
legend_bif = mlines.Line2D([], [], linestyle='None', marker='o',markerfacecolor='white', markeredgecolor='black',
    markersize=20, markeredgewidth=3)
legend_true_re = mlines.Line2D([], [], color='darkviolet', linestyle='-',linewidth=6)
legend_esti_re = mlines.Line2D([], [], linestyle='None', marker='o',markerfacecolor='white', markeredgecolor='crimson',
    markersize=20, markeredgewidth=1)

fig.legend(handles = [legend_state, legend_pstate, legend_bif, legend_true_re, legend_esti_re],
            labels = ['State','Forecast interval','Bifurcation point', 'True Re', 'Estimated Re (SINDy)'],
              loc='upper center', bbox_to_anchor=(0.5,1.01), ncol=2, frameon=False, 
              prop=font_manager.FontProperties(family='Arial', size=20, weight='normal'),handletextpad=0.4)

plt.subplots_adjust(top=0.9, bottom=0.08, left=0.15, right=0.98, hspace=0.15)
plt.savefig('../figures/SFIG3.pdf',format='pdf')
