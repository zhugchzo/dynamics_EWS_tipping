import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import numpy as np

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

df = pandas.read_csv('../data/sub_hopf/sub_hopf.csv')

df_results = pandas.read_csv('../results/sub_hopf/sub_hopf.csv')
df_stable = pandas.read_csv('../results/sub_hopf/sub_hopf_stable.csv')

df_ac_results = pandas.read_csv('../results/sub_hopf/sub_hopf_ac.csv')
df_ac_stable = pandas.read_csv('../results/sub_hopf/sub_hopf_stable_ac.csv')

df_nop_results = pandas.read_csv('../results/sub_hopf/sub_hopf_nopars.csv')
df_nop_stable = pandas.read_csv('../results/sub_hopf/sub_hopf_stable_nopars.csv')

t1 = df['Time'] + 1
t2 = df_results['Time'] + 1
t3 = df_stable['Time'] + 1

x = df['traj_x']

real = df_results['real_eval']
imag = df_results['imag_eval']

theta_x = df_results['pred_x']
ac_x = df_ac_results['pred_x']
nop_x = df_nop_results['pred_x']

theta_real = df_results['esti_real_eval1']
theta_imag1 = df_results['esti_imag_eval1']
theta_imag2 = df_results['esti_imag_eval2']
theta_stbexpnt = df_stable['stable_expnt']

ac_real = df_ac_results['esti_real_eval1']
ac_imag1 = df_ac_results['esti_imag_eval1']
ac_imag2 = df_ac_results['esti_imag_eval2']
ac_stbexpnt = df_ac_stable['stable_expnt']

nop_real = df_nop_results['esti_real_eval1']
nop_imag1 = df_nop_results['esti_imag_eval1']
nop_imag2 = df_nop_results['esti_imag_eval2']

df_nop_stable['is_nan'] = df_nop_stable['stable_expnt'].isna()
df_nop_nan = df_nop_stable[df_nop_stable['is_nan']]
df_nop = df_nop_stable[~df_nop_stable['is_nan']]

t_nop_nan = df_nop_nan['Time'] + 1
t_nop = df_nop['Time'] + 1
nop_stbexpnt_nan = np.zeros(len(t_nop_nan))
nop_stbexpnt = df_nop['stable_expnt']

subplt = ax1

subplt.scatter(t1[::5],x[::5],s=10,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,theta_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,x[1000],s=200, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-62,1265)
subplt.set_ylim(0.2,8)
subplt.set_ylabel('system state',font_y,labelpad=10)
subplt.text(820,1.44,'subcritical Hopf',size=32)
subplt.text(-0.075, 0.95,'(a)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([1,7])
subplt.tick_params(axis='x', labelbottom=False)
subplt.tick_params(axis='y', labelsize=24)

subplt.plot([15, 484], [2, 2], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 2), (494, 2), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,2.3,'sliding window', fontsize=25, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((400, 2.3), (440, 2.3),  
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)
subplt.text(900, 0.8, r'using $\theta$ for estimating', fontsize=25, ha='center', va='center', color='black')

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

ax_inset = fig.add_axes([0.16, 0.78, 0.14, 0.2]) 

ax_inset.scatter(t3, theta_stbexpnt, color='crimson', marker='x',zorder=6)
ax_inset.scatter(t3, theta_real[-30:], color='violet', marker='o',zorder=5,alpha=0.8)
ax_inset.plot(t3, theta_real[-30:], color='violet', marker='o',zorder=5,alpha=0.8)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x,labelpad=-5)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000], fontsize=18)
ax_inset.set_yticks([0,0.3])
ax_inset.set_yticklabels(['0','0.3'], fontsize=18)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.25,0.4)
#ax_inset.text(0.01, 0.85, '(a4)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
legend_re = mlines.Line2D([], [], color='violet', marker='o', linestyle='-')
ax_inset.legend(handles=[legend_index,legend_re],labels=['estimated stability index','   estimated Re'],fontsize=24,loc='center', bbox_to_anchor=(0.5,0.15), frameon=False,handletextpad=-0.1,markerscale=1.5)

#subplt.set_title('Codimension-two Subcritical Hopf Bifurcation', fontsize=30, fontweight='light', fontfamily='Times New Roman', y=1.07)
subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.25, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':24})

subplt = ax2

subplt.scatter(t2,real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2,theta_real,s=20,c='crimson',marker='^')
subplt.scatter(1000,real[499],s=200,marker='^',facecolors='white', edgecolors='black')
subplt.text(1020,0,'reach zero',size=24)
subplt.set_xlabel('Time',font_x)
subplt.set_ylabel('Re',font_y,labelpad=-25)
subplt.set_xlim(-62,1265)
subplt.text(-0.075, 0.95,'(d)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_ylim(-0.25,0.02)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([-0.2,0])
subplt.tick_params(axis='x', labelsize=24)
subplt.set_yticklabels(['-0.2','0'], fontsize=24)

ax_inset = fig.add_axes([0.035, 0.17, 0.105, 0.22]) 

ax_inset.scatter(t2[::5],theta_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',label = 'estimated Im')
ax_inset.scatter(t2[::5],theta_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson')
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
ax_inset.legend(fontsize=22,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.2),handletextpad=0,markerscale=1.5)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=5, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center',frameon=False, bbox_to_anchor=(0.8,0.25), handletextpad=0,markerscale=2.5,prop={'size':24})

subplt = ax3

subplt.scatter(t1[::5],x[::5],s=10,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,ac_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,x[1000],s=200, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-62,1265)
subplt.set_ylim(0.2,8)
subplt.text(820,1.44,'subcritical Hopf',size=32)
subplt.text(-0.075, 0.95,'(b)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([1,7])
subplt.tick_params(axis='x', labelbottom=False)
subplt.tick_params(axis='y', labelsize=24)

subplt.plot([15, 484], [2, 2], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 2), (494, 2), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,2.3,'sliding window', fontsize=25, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((400, 2.3), (440, 2.3), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)
subplt.text(900, 0.8,r'using parameters $a$ and $c$ for estimating', fontsize=20, ha='center', va='center', color='black')

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

ax_inset = fig.add_axes([0.49, 0.78, 0.14, 0.2]) 

ax_inset.scatter(t3, ac_stbexpnt, color='crimson', marker='x',zorder=6)
ax_inset.scatter(t3, ac_real[-30:], color='violet', marker='o',zorder=5,alpha=0.8)
ax_inset.plot(t3, ac_real[-30:], color='violet', marker='o',zorder=5,alpha=0.8)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x,labelpad=-5)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000], fontsize=18)
ax_inset.set_yticks([0,0.3])
ax_inset.set_yticklabels(['0','0.3'], fontsize=18)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.25,0.4)
#ax_inset.text(0.01, 0.85, '(b4)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
legend_re = mlines.Line2D([], [], color='violet', marker='o', linestyle='-')
ax_inset.legend(handles=[legend_index,legend_re],labels=['estimated stability index','   estimated Re'],fontsize=22,loc='center', bbox_to_anchor=(0.5,0.12), frameon=False,handletextpad=-0.1,markerscale=1.5)

#subplt.set_title('Codimension-two Subcritical Hopf Bifurcation', fontsize=30, fontweight='light', fontfamily='Times New Roman', y=1.07)
subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.25, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':24})

subplt = ax4

subplt.scatter(t2,real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2,ac_real,s=20,c='crimson',marker='^')
subplt.scatter(1000,real[499],s=200,marker='^',facecolors='white', edgecolors='black')
subplt.text(1020,0,'reach zero',size=24)
subplt.set_xlim(-62,1265)
subplt.text(-0.075, 0.95,'(e)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_ylim(-0.25,0.02)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([-0.2,0])
subplt.set_xlabel('Time',font_x)
subplt.tick_params(axis='x', labelsize=24)
subplt.set_yticklabels(['-0.2','0'], fontsize=24)

ax_inset = fig.add_axes([0.365, 0.17, 0.105, 0.22])

ax_inset.scatter(t2[::5],ac_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',label = 'estimated Im')
ax_inset.scatter(t2[::5],ac_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson')
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
#ax_inset.text(0.04, 0.85, '(b3)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
ax_inset.legend(fontsize=22,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.2),handletextpad=0,markerscale=1.5)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=5, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center',frameon=False, bbox_to_anchor=(0.8,0.25), handletextpad=0,markerscale=2.5,prop={'size':24})

subplt = ax5

subplt.scatter(t1[::5],x[::5],s=10,c='darkslateblue',marker='o')
subplt.plot(t1[::5],x[::5],c='darkslateblue')
subplt.scatter(t2[::5]+1,nop_x[::5],s=100, marker='o',facecolors='none',edgecolors='crimson')
subplt.scatter(1000,x[1000],s=200, marker='o',facecolors='white', edgecolors='black',zorder=5)
subplt.set_xlim(-62,1265)
subplt.set_ylim(0.2,8)
subplt.text(820,1.44,'subcritical Hopf',size=32)
subplt.text(-0.075, 0.95,'(c)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=0, color='black', linestyle='--', linewidth=2)
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([1,7])
subplt.tick_params(axis='x', labelbottom=False)
subplt.tick_params(axis='y', labelsize=24)

subplt.plot([15, 484], [2, 2], linestyle='--', color='black', linewidth=2)
arrow1 = FancyArrowPatch((5, 2), (494, 2), 
                        arrowstyle='<|-|>', 
                        color='black',
                        linewidth = 0, 
                        mutation_scale=25)
subplt.add_patch(arrow1)

subplt.text(240,2.3,'sliding window', fontsize=25, ha='center', va='center', color='black')
arrow2 = FancyArrowPatch((400, 2.3), (440, 2.3), 
                        arrowstyle='->', 
                        color='black',
                        linewidth=2,
                        mutation_scale=15)
subplt.add_patch(arrow2)
subplt.text(900, 0.8,'estimating without parameters', fontsize=25, ha='center', va='center', color='black')

legend_state = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=3, linestyle='-', markeredgewidth=1.5)
legend_pstate = mlines.Line2D([], [], markerfacecolor='none',color='crimson', marker='o', markersize=5, linestyle='None', markeredgewidth=1.5)

ax_inset = fig.add_axes([0.82, 0.78, 0.14, 0.2]) 

ax_inset.scatter(t_nop, nop_stbexpnt, color='crimson', marker='x',zorder=6)
ax_inset.scatter(t_nop_nan, nop_stbexpnt_nan-0.045, color='cyan', marker='x',zorder=6)
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x,labelpad=-5)
ax_inset.set_xticks([971,1000])
ax_inset.set_xticklabels([971,1000], fontsize=18)
ax_inset.set_yticks([0])
ax_inset.set_yticklabels(['0'], fontsize=18)
ax_inset.yaxis.tick_right()
ax_inset.set_ylim(-0.048,0.03)
#ax_inset.text(0.01, 0.85, '(b4)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
legend_index = mlines.Line2D([], [], color='crimson', marker='x', linestyle='None',markeredgewidth=2)
legend_nan = mlines.Line2D([], [], color='cyan', marker='x', linestyle='None',markeredgewidth=2)
ax_inset.legend(handles=[legend_index,legend_nan],labels=['estimated stability index','Nan'],fontsize=22,loc='center', bbox_to_anchor=(0.5,0.8), frameon=False,handletextpad=-0.1,markerscale=1.5)

#subplt.set_title('Codimension-two Subcritical Hopf Bifurcation', fontsize=30, fontweight='light', fontfamily='Times New Roman', y=1.07)
subplt.legend(handles=[legend_state,legend_pstate],labels=['state','predicted state'],loc='upper center', frameon=False, bbox_to_anchor=(0.25, 0.8), handletextpad=0.2,markerscale=2.5,prop={'size':24})

subplt = ax6

subplt.scatter(t2,real,s=20,c='darkslateblue',marker='^')
subplt.scatter(t2,nop_real,s=20,c='crimson',marker='^')
subplt.scatter(1000,real[499],s=200,marker='^',facecolors='white', edgecolors='black')
subplt.text(1020,0,'reach zero',size=24)
subplt.set_xlim(-62,1265)
subplt.text(-0.075, 0.95,'(f)',ha='left', transform=subplt.transAxes,fontdict={'family':'Times New Roman','size':35,'weight':'light'})
subplt.axvline(x=500, color='black', linestyle='--', linewidth=2)
subplt.set_ylim(-0.25,0.06)
subplt.set_xticks([0,500,1000])
subplt.set_yticks([-0.2,0])
subplt.set_xlabel('Time',font_x)
subplt.tick_params(axis='x', labelsize=24)
subplt.set_yticklabels(['-0.2','0'], fontsize=24)

ax_inset = fig.add_axes([0.695, 0.17, 0.105, 0.22])

ax_inset.scatter(t2[::5],nop_imag1[::5], s=80, marker='^',facecolors='none',edgecolors='crimson',label = 'estimated Im')
ax_inset.scatter(t2[::5],nop_imag2[::5], s=80, marker='^',facecolors='none',edgecolors='crimson')
ax_inset.axhline(y=0, color='black', linestyle='--')
ax_inset.set_xlabel('Time',font_in_x,labelpad=-5)
ax_inset.set_ylabel('Im',font_in_y,labelpad=20)
ax_inset.set_xticks([500,1000])
ax_inset.set_xticklabels([500,1000], fontsize=18)
ax_inset.set_yticks([-0.1,0,0.1])
ax_inset.set_yticklabels([-0.1,0,0.1], fontsize=18)
ax_inset.set_ylim(-0.2,0.2)
ax_inset.yaxis.tick_right()
ax_inset.yaxis.set_label_position("right") 
ax_inset.yaxis.set_label_coords(1.07, 0.96)
ax_inset.yaxis.label.set_rotation(360)
#ax_inset.text(0.04, 0.85, '(b3)', ha='left', va='bottom', transform=ax_inset.transAxes, fontdict={'family':'Times New Roman', 'size':35, 'weight':'light'})
ax_inset.legend(fontsize=22,frameon=False,loc='upper center', bbox_to_anchor=(0.5,0.2),handletextpad=0,markerscale=1.5)

legend_gt = mlines.Line2D([], [], color='darkslateblue', marker='^', markersize=5, linestyle='None')
legend_pred = mlines.Line2D([], [], color='crimson', marker='^', markersize=5, linestyle='None')

subplt.legend(handles=[legend_gt,legend_pred],labels=['ground truth','estimated Re'],loc='upper center',frameon=False, bbox_to_anchor=(0.8,0.25), handletextpad=0,markerscale=2.5,prop={'size':24})

plt.subplots_adjust(top=0.99, bottom=0.055, left=0.03, right=0.99, hspace=0.05, wspace=0.1)
plt.savefig('../figures/SFIG1.pdf',format='pdf',dpi=600)