import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

font_x = {'family':'Times New Roman','weight':'light','size': 43}
font_y = {'family':'Times New Roman','weight':'light','size': 52}

fig, axs = plt.subplots(2, 2, figsize=(24,15))

(ax1, ax3, ax2, ax4) = axs.flatten()

df_fold = pandas.read_csv('../data/fold/fold.csv')
df_fold_results = pandas.read_csv('../results/fold/fold.csv')
df_nobif = pandas.read_csv('../data/no_bif/no_bif_train.csv')
df_nobif_results = pandas.read_csv('../results/no_bif/no_bif.csv')

t1 = df_fold['Time'] + 1
t2 = df_fold_results['Time'] + 1
t3 = df_nobif['Time'] + 1
t4 = df_nobif_results['Time'] + 1

fold_x = df_fold['traj_x']
fold_pred_x = df_fold_results['pred_x']
nobif_x = df_nobif['traj_x']
nobif_pred_x = df_nobif_results['pred_x']

fold_rrate = df_fold_results['rrate']
fold_esti_rrate = df_fold_results['esti_rrate']
nobif_rrate = df_nobif_results['rrate']
nobif_esti_rrate = df_nobif_results['esti_rrate']

subplt = ax1

subplt.scatter(t1[::5],fold_x[::5],s=13,c='darkslateblue',marker='o')
subplt.plot(t1[::5],fold_x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,fold_pred_x[::5],s=130, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,fold_x[1000],s=280, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_ylabel('system state',font_y,labelpad=-5)
subplt.set_xlim(-55,1100)
subplt.set_xticks([0,500,1000])
subplt.set_ylim(-0.1,8.5)
subplt.set_yticks([1,7])
subplt.tick_params(axis='x', labelbottom=False)
subplt.set_yticklabels(['1','7'], fontsize=32)
subplt.text(820,3.7,'fold',size=48)
subplt.text(-0.075, 0.95,'(a)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)

subplt.plot([15, 484], [6, 6], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 6), (494, 6), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,5.5,'sliding window', fontsize=32, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((420, 5.5), (460, 5.5), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

#subplt.set_title('Codimension-one Fold Bifurcation', fontsize=30, fontweight='light', fontfamily='Times New Roman', y=1.07)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='lower center', frameon=False, bbox_to_anchor=(0.7, 0.1), handletextpad=0.2, markerscale=2.5,prop={'size':32})

subplt = ax2

subplt.scatter(t2,fold_rrate,s=28,c='darkslateblue',marker='^')
subplt.scatter(t2,fold_esti_rrate,s=28,c='crimson',marker='^')
subplt.scatter(1000,fold_rrate[499],s=200,marker='^',facecolors='white', edgecolors='black')
subplt.text(730,-0.02,'reach zero',size=32)
subplt.set_xlabel('Time',font_x)
subplt.set_ylabel('Re',font_y,labelpad=-55)
subplt.set_xlim(-55,1100)
subplt.set_xticks([0,500,1000])
subplt.set_ylim(-0.55,0.05)
subplt.set_yticks([-0.5,0])
subplt.tick_params(axis='x', labelsize=32)
subplt.set_yticklabels(['-0.5','0'], fontsize=32)
subplt.text(-0.075, 0.96,'(c)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=5, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center', frameon=False, bbox_to_anchor=(0.25,0.6), handletextpad=0, markerscale=2.5,prop={'size':32})

subplt = ax3

subplt.scatter(t3[::5],nobif_x[::5],s=13,c='darkslateblue',marker='o')
subplt.plot(t3[::5],nobif_x[::5],c='darkslateblue')
subplt.scatter(t4[::5]+1,nobif_pred_x[::5],s=130, marker='o',facecolors='none',edgecolors='crimson')
subplt.set_xlim(-55,1055)
subplt.set_xticks([0,500,1000])
subplt.set_ylim(0.25,1.05)
subplt.set_yticks([0.3,0.9])
subplt.tick_params(axis='x', labelbottom=False)
subplt.set_yticklabels(['0.3','0.9'], fontsize=32)
subplt.text(-0.075, 0.95,'(b)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.text(600,0.5,'no transition',size=48)

subplt.plot([15, 484], [0.4, 0.4], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 0.4), (494, 0.4), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,0.35,'sliding window', fontsize=32, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((420, 0.35), (460, 0.35), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

#subplt.set_title('No Tipping Point', fontsize=30, fontweight='light', fontfamily='Times New Roman', y=1.07)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='lower center',frameon=False, bbox_to_anchor=(0.75, 0.7), handletextpad=0.2, markerscale=2.5,prop={'size':32})

subplt = ax4

subplt.scatter(t4,nobif_rrate,s=28,c='darkslateblue',marker='^')
subplt.scatter(t4,nobif_esti_rrate,s=28,c='crimson',marker='^')
subplt.set_xlabel('Time',font_x)
subplt.set_xlim(-55,1055)
subplt.set_xticks([0,500,1000])
subplt.set_ylim(-1,-0.4)
subplt.set_yticks([-0.9,-0.5])
subplt.tick_params(axis='x', labelsize=32)
subplt.set_yticklabels(['-0.9','-0.5'], fontsize=32)
subplt.text(-0.075, 0.96,'(d)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=5, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center',frameon=False, bbox_to_anchor=(0.25,0.6), handletextpad=0, markerscale=2.5,prop={'size':32})

plt.subplots_adjust(top=0.99, bottom=0.068, left=0.04, right=0.99, hspace=0.05, wspace=0.1)
plt.savefig('../figures/FIG1.pdf',format='pdf',dpi=600)
