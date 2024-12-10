import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)

# Change the working directory to the directory of the current file
os.chdir(current_dir)

font_x = {'family':'Times New Roman','weight':'light','size': 32}
font_y = {'family':'Times New Roman','weight':'light','size': 40}
font_in_x = {'family':'Times New Roman','weight':'light','size': 28}
font_in_y = {'family':'Times New Roman','weight':'light','size': 30}

fig, axs = plt.subplots(2, 3, figsize=(36,15))

(ax1, ax3, ax5, ax2, ax4, ax6) = axs.flatten()

df_sup_hopf = pandas.read_csv('../data/sup_hopf/sup_hopf.csv')
df_sup_hopf_results = pandas.read_csv('../results/sup_hopf/sup_hopf.csv')
df_sup_hopf_stable = pandas.read_csv('../results/sup_hopf/sup_hopf_stable.csv')

df_trans = pandas.read_csv('../data/trans/trans.csv')
df_trans_results = pandas.read_csv('../results/trans/trans.csv')

df_sub_hopf = pandas.read_csv('../data/sub_hopf/sub_hopf.csv')
df_sub_hopf_results = pandas.read_csv('../results/sub_hopf/sub_hopf.csv')
df_sub_hopf_stable = pandas.read_csv('../results/sub_hopf/sub_hopf_stable.csv')

t1 = df_sup_hopf['Time'] + 1
t2 = df_sup_hopf_results['Time'] + 1
t3 = df_sup_hopf_stable['Time'] + 1

t4 = df_trans['Time'] + 1
t5 = df_trans_results['Time'] + 1

t6 = df_sub_hopf['Time'] + 1
t7 = df_sub_hopf_results['Time'] + 1
t8 = df_sub_hopf_stable['Time'] + 1

sup_hopf_x = df_sup_hopf['traj_x']
sup_hopf_pred_x = df_sup_hopf_results['pred_x']

trans_x = df_trans['traj_x']
trans_pred_x = df_trans_results['pred_x']

sub_hopf_x = df_sub_hopf['traj_x']
sub_hopf_pred_x = df_sub_hopf_results['pred_x']

sup_hopf_real = df_sup_hopf_results['real_eval']
sup_hopf_esti_real = df_sup_hopf_results['esti_real_eval1']
sup_hopf_imag = df_sup_hopf_results['imag_eval']
sup_hopf_esti_imag1 = df_sup_hopf_results['esti_imag_eval1']
sup_hopf_esti_imag2 = df_sup_hopf_results['esti_imag_eval2']
sup_hopf_esti_stbexpnt = df_sup_hopf_stable['stable_expnt']

trans_real = df_trans_results['real_eval']
trans_esti_real = df_trans_results['esti_real_eval1']
trans_imag = df_trans_results['imag_eval']
trans_esti_imag1 = df_trans_results['esti_imag_eval1']

sub_hopf_real = df_sub_hopf_results['real_eval']
sub_hopf_esti_real = df_sub_hopf_results['esti_real_eval1']
sub_hopf_imag = df_sub_hopf_results['imag_eval']
sub_hopf_esti_imag1 = df_sub_hopf_results['esti_imag_eval1']
sub_hopf_esti_imag2 = df_sub_hopf_results['esti_imag_eval2']
sub_hopf_esti_stbexpnt = df_sub_hopf_stable['stable_expnt']

subplt = ax1

subplt.scatter(t1[::5],sup_hopf_x[::5],s=10,c='darkslateblue',marker='o')
subplt.plot(t1[::5],sup_hopf_x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,sup_hopf_pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,sup_hopf_x[1000],s=200, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_ylabel('system state',font_y,labelpad=-5)
subplt.set_xlim(-65,1326)
subplt.set_ylim(0.5,2.2)
subplt.text(650,0.7,'supercritical Hopf',size=36)
subplt.text(-0.075, 0.95,'(a)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.tick_params(axis='x', labelbottom=False)
subplt.set_yticks([0.8, 2])
subplt.set_yticklabels(['0.8','2'], fontsize=24)

subplt.plot([15, 484], [0.9, 0.9], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 0.9), (494, 0.9), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,0.8,'sliding window', fontsize=25, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((420, 0.8), (460, 0.8), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

ax_inset = fig.add_axes([0.1575, 0.78, 0.145, 0.2]) 

ax_inset.scatter(t3, sup_hopf_esti_stbexpnt, color='crimson', marker='x')
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x,labelpad=-5)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000], fontsize=18)
ax_inset.set_yticks([-0.2,0])
ax_inset.set_yticklabels(['-0.2','0'], fontsize=18)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.23,0.046)
#ax_inset.text(0.01, 0.85, '(a4)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
ax_inset.legend(handles=[legend_index],labels=['estimated stability index'],fontsize=25,loc='center', bbox_to_anchor=(0.5,0.91), frameon=False,handletextpad=-0.1,markerscale=1.5)

#subplt.set_title('Codim-1 Supercritical Hopf Bifurcation', fontsize=40, fontweight='light', fontfamily='Times New Roman', y=1.07)
subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.24, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':24})

subplt = ax2

subplt.scatter(t2,sup_hopf_real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2[df_sup_hopf_results['esti_real_eval1'].between(-0.065, 0.002)],sup_hopf_esti_real[df_sup_hopf_results['esti_real_eval1'].between(-0.065, 0.002)],s=20,c='crimson',marker='^')
subplt.scatter(1000,sup_hopf_real[499],s=200,marker='^',facecolors='white', edgecolors='black')
subplt.text(1020,0,'reach zero',size=24)
subplt.set_xlabel('Time',font_x)
subplt.set_ylabel('Re',font_y,labelpad=-35)
subplt.text(-0.075, 0.95,'(d)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xlim(-65,1326)
subplt.set_ylim(-0.066,0.006)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([-0.05,0])
subplt.tick_params(axis='x', labelsize=24)
subplt.set_yticklabels(['-0.05','0'], fontsize=24)

ax_inset = fig.add_axes([0.035, 0.17, 0.095, 0.22]) 

ax_inset.scatter(t2[::5],sup_hopf_esti_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',label = 'estimated Im')
ax_inset.scatter(t2[::5],sup_hopf_esti_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson')
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x,labelpad=-5)
ax_inset.set_ylabel('Im',font_in_y,labelpad=30)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000], fontsize=18)
ax_inset.set_yticks([-0.2,0,0.2])
ax_inset.set_yticklabels([-0.2,0,0.2], fontsize=18)
ax_inset.set_ylim(-0.4,0.4)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right") 
ax_inset.yaxis.set_label_coords(1.07, 0.96)
ax_inset.yaxis.label.set_rotation(360)
#ax_inset.text(0.04, 0.85, '(a3)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
ax_inset.legend(fontsize=25,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.3),handletextpad=0,markerscale=1.5)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=5, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center', frameon=False, bbox_to_anchor=(0.8, 0.25),handletextpad=0,markerscale=2.5,prop={'size':24})

subplt = ax3

subplt.scatter(t4[::5],trans_x[::5],s=10,c='darkslateblue',marker='o')
subplt.plot(t4[::5],trans_x[::5],c='darkslateblue')
subplt.scatter(t5[::5]+1,trans_pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,trans_x[1000],s=200, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-55,1100)
subplt.set_ylim(1.65,1.96)
subplt.set_yticks([1.7,1.9])
subplt.set_yticklabels(['1.7','1.9'], fontsize=24)
subplt.text(680,1.72,'transcritical',size=36)
subplt.text(-0.075, 0.95,'(b)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.tick_params(axis='x', labelbottom=False)

subplt.plot([15, 484], [1.72, 1.72], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 1.72), (494, 1.72), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,1.735,'sliding window', fontsize=25, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((400, 1.735), (440, 1.735), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

#subplt.set_title('Codim-1 Transcritical Bifurcation', fontsize=40, fontweight='light', fontfamily='Times New Roman', y=1.07)
subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.75, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':24})

subplt = ax4

subplt.scatter(t5,trans_real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t5,trans_esti_real,s=20,c='crimson',marker='^')
subplt.scatter(1000,trans_real[499],s=200,marker='^',facecolors='white', edgecolors='black')
subplt.text(800,0.01,'reach zero',size=24)
subplt.set_xlabel('Time',font_x)
subplt.set_xlim(-55,1100)
subplt.text(-0.075, 0.95,'(e)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([-0.5,0])
subplt.tick_params(axis='x', labelsize=24)
subplt.set_yticklabels(['-0.5','0'], fontsize=24)

ax_inset = fig.add_axes([0.375, 0.17, 0.11, 0.22])

ax_inset.scatter(t5[::5],trans_esti_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',label = 'estimated Im')
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x,labelpad=-5)
ax_inset.set_ylabel('Im',font_in_y,labelpad=20)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000], fontsize=18)
ax_inset.set_yticks([0])
ax_inset.set_yticklabels([0], fontsize=18)
ax_inset.set_ylim(-1.1,1.1)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right") 
ax_inset.yaxis.set_label_coords(1.07, 0.96)
ax_inset.yaxis.label.set_rotation(360)
#ax_inset.text(0.04, 0.85, '(b3)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
ax_inset.legend(fontsize=25,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.3),handletextpad=0,markerscale=1.5)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=5, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center', frameon=False, bbox_to_anchor=(0.8,0.25),handletextpad=0,markerscale=2.5,prop={'size':24})

subplt = ax5

subplt.scatter(t6[::5],sub_hopf_x[::5],s=10,c='darkslateblue',marker='o')
subplt.plot(t6[::5],sub_hopf_x[::5],c='darkslateblue')
subplt.scatter(t7[::5]+1,sub_hopf_pred_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,sub_hopf_x[1000],s=200, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-62,1265)
subplt.set_ylim(0,10)
subplt.text(750,1.2,'subcritical Hopf',size=36)
subplt.text(-0.075, 0.95,'(c)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([1,9])
subplt.tick_params(axis='x', labelbottom=False)
subplt.tick_params(axis='y', labelsize=24)

subplt.plot([15, 484], [2, 2], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 2), (494, 2), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,2.5,'sliding window', fontsize=25, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((400, 2.5), (440, 2.5), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)

ax_inset = fig.add_axes([0.822, 0.78, 0.145, 0.2]) 

ax_inset.scatter(t8, sub_hopf_esti_stbexpnt, color='crimson', marker='x')
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x,labelpad=-5)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000], fontsize=18)
ax_inset.set_yticks([0,0.3])
ax_inset.set_yticklabels(['0','0.3'], fontsize=18)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.08,0.4)
#ax_inset.text(0.01, 0.85, '(a4)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
ax_inset.legend(handles=[legend_index],labels=['estimated stability index'],fontsize=24,loc='center',bbox_to_anchor=(0.5,0.09), frameon=False,handletextpad=-0.1,markerscale=1.5)

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

#subplt.set_title('Codimension-two Subcritical Hopf Bifurcation', fontsize=30, fontweight='light', fontfamily='Times New Roman', y=1.07)
subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.25, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':24})

subplt = ax6

subplt.scatter(t7,sub_hopf_real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t7,sub_hopf_esti_real,s=20,c='crimson',marker='^')
subplt.scatter(1000,sub_hopf_real[499],s=200,marker='^',facecolors='white', edgecolors='black')
subplt.text(1020,0,'reach zero',size=24)
subplt.set_xlabel('Time',font_x)
subplt.set_xlim(-62,1265)
subplt.text(-0.075, 0.95,'(f)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_ylim(-0.25,0.02)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([-0.2,0])
subplt.tick_params(axis='x', labelsize=24)
subplt.set_yticklabels(['-0.2','0'], fontsize=24)

ax_inset = fig.add_axes([0.695, 0.17, 0.1, 0.22]) 

ax_inset.scatter(t7[::5],sub_hopf_esti_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson', label = 'estimated Im')
ax_inset.scatter(t7[::5],sub_hopf_esti_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson')
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x,labelpad=-5)
ax_inset.set_ylabel('Im',font_in_y,labelpad=20)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000], fontsize=18)
ax_inset.set_yticks([-0.3,0,0.3])
ax_inset.set_yticklabels([-0.3,0,0.3], fontsize=18)
ax_inset.set_ylim(-0.6,0.6)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right") 
ax_inset.yaxis.set_label_coords(1.07, 0.96)
ax_inset.yaxis.label.set_rotation(360)
#ax_inset.text(0.04, 0.85, '(a3)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
ax_inset.legend(fontsize=25,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.2),handletextpad=0,markerscale=1.5)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=5, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center',frameon=False, bbox_to_anchor=(0.8,0.25), handletextpad=0,markerscale=2.5,prop={'size':24})

plt.subplots_adjust(top=0.99, bottom=0.055, left=0.03, right=0.99, hspace=0.05, wspace=0.1)
plt.savefig('../figures/FIG3.pdf',format='pdf',dpi=600)
