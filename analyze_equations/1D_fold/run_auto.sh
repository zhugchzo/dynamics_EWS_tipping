#!/bin/bash

# output equi.csv, pars.csv
echo Run fold.py
/opt/miniconda3/envs/comp/bin/python /Users/zhugchzo/Desktop/2paper_code/analyze_equations/1D_fold/fold.py

# Run bifurcation continuation using AUTO and output b.out files for each varied parameter
# (Make sure AUTO runs using Python 2)
echo Run run_cont.auto
auto /Users/zhugchzo/Desktop/2paper_code/analyze_equations/1D_fold/run_cont.auto

# Output bifurcations.
echo Run combine_auto_output.py
/opt/miniconda3/envs/comp/bin/python /Users/zhugchzo/Desktop/2paper_code/analyze_equations/1D_fold/combine_auto_output.py

rm -r output_auto

