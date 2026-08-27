import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import font_manager

font_x = {'family':'Arial','weight':'normal','size': 60}
font_y = {'family':'Arial','weight':'normal','size': 60}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 45
plt.rcParams['ytick.labelsize'] = 40

fig, axs = plt.subplots(2, 1, figsize=(10,16))

###########################################################################################

axs[0].axis('off')

###########################################################################################

df_fold_BA = pd.read_csv('../data/fold_BA/fold_BA.csv')

plot_begin = 995
plot_over = 1027

transient_begin = 999
transient_over = 1023

t_plot = df_fold_BA['Time'][plot_begin:plot_over] + 1
t_transient = df_fold_BA['Time'][transient_begin:transient_over] + 1
traj_mean = df_fold_BA[[f'traj_x{i}' for i in range(20)]].mean(axis=1)

ax_transient = axs[1]

ax_transient.scatter(t_plot,traj_mean[plot_begin:plot_over],s=150,c='darkslateblue',marker='o')
ax_transient.plot(t_plot,traj_mean[plot_begin:plot_over],c='darkslateblue',linewidth=6)
ax_transient.scatter(t_transient,traj_mean[transient_begin:transient_over],s=150,c='darkgrey',marker='o',zorder=4)
ax_transient.plot(t_transient,traj_mean[transient_begin:transient_over],c='darkgrey',linewidth=6)
ax_transient.scatter(1000,traj_mean[1000],s=1000, marker='o',facecolors='white', edgecolors='black',linewidths=3,zorder=5)

ax_transient.set_yscale('log')

ax_transient.set_xlabel('Time',font_x,labelpad=5)
ax_transient.set_ylabel('System state',font_y)
ax_transient.yaxis.set_label_coords(-0.15, 0.5)

ax_transient.set_xlim(993,1029)
ax_transient.set_xticks([1000,1023])
ax_transient.set_xticklabels(['1000','1023'])
ax_transient.set_ylim(0.05,15)
ax_transient.set_yticks([0.1, 1, 10])
ax_transient.tick_params(direction='in', length=10)

ax_transient.minorticks_on()
ax_transient.tick_params(axis='x', which='minor', bottom=False)
ax_transient.tick_params(axis='y', which='minor', direction='in', length=5)

ax_transient.set_title('BA network', fontsize=60, fontweight='normal', fontfamily='DejaVu Sans', fontstyle= 'italic', y=1.05)
ax_transient.text(-0.25,1.05,'(b)',ha='left', transform=ax_transient.transAxes,fontdict={'family':'Arial','size':80,'weight':'normal'})

plt.subplots_adjust(top=0.99, bottom=0.1, left=0.2, right=0.95, hspace=0.15)
plt.savefig('../figures/SFIG4.3.pdf',format='pdf')