import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines

font_x = {'family':'Arial','weight':'normal','size': 40}
font_y = {'family':'Arial','weight':'normal','size': 40}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 36
plt.rcParams['ytick.labelsize'] = 36

fig, axs = plt.subplots(2, 2, figsize=(24,15))

(ax1, ax3, ax2, ax4) = axs.flatten()

df = pandas.read_csv('../data/fold/fold.csv')
df_sindy_results = pandas.read_csv('../results/fold/SINDy_fold.csv')
df_mlp_results = pandas.read_csv('../results/fold/MLP_nonlin_fold.csv')

t1 = df['Time'] + 1
t2 = df_sindy_results['Time'] + 1
t3 = df_mlp_results['Time'] + 1

x = df['traj_x']
sindy_pred_x = df_sindy_results['pred_x']
mlp_pred_x = df_mlp_results['pred_x']

rrate = df_sindy_results['rrate']
sindy_esti_rrate = df_sindy_results['esti_rrate']
mlp_esti_rrate = df_mlp_results['esti_rrate']

subplt = ax1

subplt.scatter(t1[::5],x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,sindy_pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,x[1000],s=500, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_ylabel('system state',font_y,labelpad=-5)
subplt.set_xlim(-55,1100)
subplt.set_xticks([0,1000])
subplt.set_ylim(-0.1,8.5)
subplt.set_yticks([1,7])
subplt.tick_params(axis='x', labelbottom=False)
subplt.set_yticklabels(['1','7'])
subplt.text(820,3.7,'fold',size=48)
subplt.text(-0.075, 0.96,'a',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)

subplt.plot([15, 484], [6, 6], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 6), (494, 6), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,5.5,'sliding window', fontsize=36, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((430, 5.5), (460, 5.5), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state\n    (SINDy)'],loc='lower center', frameon=False, bbox_to_anchor=(0.71, 0.05), handletextpad=0.2, markerscale=2.5,prop={'size':36})

subplt = ax2

subplt.scatter(t2,rrate,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2,sindy_esti_rrate,s=20,c='crimson',marker='^')
subplt.scatter(1000,rrate[499],s=500,marker='^',facecolors='white', edgecolors='black')
subplt.text(680,-0.02,'reach zero',size=40)
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_ylabel('Re',font_y,labelpad=-55)
subplt.set_xlim(-55,1100)
subplt.set_xticks([0,1000])
subplt.set_ylim(-0.55,0.05)
subplt.set_yticks([-0.5,0])
subplt.set_yticklabels(['-0.5','0'])
subplt.text(-0.075, 0.96,'c',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=10, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=10, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re\n    (SINDy)'],loc='upper center', frameon=False, bbox_to_anchor=(0.25,0.6), handletextpad=0, markerscale=2.5,prop={'size':36})

subplt = ax3

subplt.scatter(t1[::5],x[::5],s=25,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t3[::5]+1,mlp_pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,x[1000],s=500, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-55,1100)
subplt.set_xticks([0,1000])
subplt.set_ylim(-0.1,8.5)
subplt.set_yticks([1,7])
subplt.tick_params(axis='x', labelbottom=False)
subplt.set_yticklabels(['1','7'])
subplt.text(820,3.7,'fold',size=48)
subplt.text(-0.075, 0.93,'b',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)

subplt.plot([15, 484], [6, 6], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 6), (494, 6), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,5.5,'sliding window', fontsize=36, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((430, 5.5), (460, 5.5), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=5, linestyle='-', markeredgewidth=1)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=10, linestyle='None', markeredgewidth=3)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state\n    (MLP)'],loc='lower center', frameon=False, bbox_to_anchor=(0.7, 0.05), handletextpad=0.2, markerscale=2.5,prop={'size':36})

subplt = ax4

subplt.scatter(t3,rrate,s=20,c='darkslateblue',marker='^')
subplt.scatter(t3,mlp_esti_rrate,s=20,c='crimson',marker='^')
subplt.scatter(1000,rrate[499],s=500,marker='^',facecolors='white', edgecolors='black')
subplt.text(680,-0.02,'reach zero',size=40)
subplt.set_xlabel('Timepoint',font_x,labelpad=-20)
subplt.set_xlim(-55,1100)
subplt.set_xticks([0,1000])
subplt.set_ylim(-0.55,0.05)
subplt.set_yticks([-0.5,0])
subplt.set_yticklabels(['-0.5','0'])
subplt.text(-0.075, 0.96,'d',ha='left', transform=subplt.transAxes,fontdict={'family':'DejaVu Sans','size':50,'weight':'bold'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=10, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=10, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re\n    (MLP)'],loc='upper center', frameon=False, bbox_to_anchor=(0.25,0.6), handletextpad=0, markerscale=2.5,prop={'size':36})

plt.subplots_adjust(top=0.98, bottom=0.06, left=0.04, right=0.99, hspace=0.05, wspace=0.15)
plt.savefig('../figures/FIG2.pdf',format='pdf',dpi=600)
