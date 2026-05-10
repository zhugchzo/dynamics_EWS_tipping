# Detecting Regime Shifts from Dynamics Using Early Warning Signals

This is the code repository to accompnay the article:

***Detecting Regime Shifts from Dynamics Using Early Warning Signals.*** *Chengzuo Zhuge, Ziting Luo, Wei Chen.*

# Requirements

Python 3.12 is required. To install python package dependencies, use the command

``` setup
pip install -r requirements.txt
```

within a new virtual environment.

# Directories

**./analyze_equations:** Code to perform bifurcation analysis on reconstructed equations.

**./compute_ROC:** Code for computing early warning indicators using the dynamics-based framework and two competing approaches on synthetic systems, as well as the metrics required for plotting ROC curves.

**./compute_mean_lambda:** Code to compute the average dynamics-based early warning signals on synthetic systems.

**./data:** Data used in this manuscript, including the synthetic data and the empirical data.

**./draw_fig:** Code to generate figures used in manuscript and supplementary material.

**./empirical_test:** Code for computing dynamics-based early warning signals on two real-world systems.

**./figures:** Figures used in manuscript and supplementary material.

**./other algorithms:** Code to compute early warning signals by the dynamics-based framework with SINDy.

**./results:** Experimental results.

**./robust_noise:** Code for noise robustness test in supplementary material.

**./sim_data:** Code to generate the time series for synthetic systems.

# Data sources

The empirical data used in this study are available from the following sources:

1. **Cyanobacterial population collapse** data is availalble in the folder `./data/microcosm`. This dataset has been cleaned. The raw data was collected by AJ Veraart et al, and was first published in [Veraart A J, Faassen E J, Dakos V, et al. Recovery rates reflect distance to a tipping point in a living system[J]. Nature, 2012, 481(7381): 357-359.] (https://www.nature.com/articles/nature10723)

2. **Fish school transition** data is availalble in the folder `./data/fish_school`. This dataset has been cleaned. The raw data was collected by Guozheng Lin et al, and was first published in [Lin G, Escobedo R, Li X, et al. Experimental evidence of stress-induced critical state in schooling fish[J]. PRX Life, 2025, 3(3): 033018.] (https://journals.aps.org/prxlife/abstract/10.1103/nr7p-m4ff)



