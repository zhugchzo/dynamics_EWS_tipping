#!/bin/bash

# Run bifurcation continuation using AUTO and output b.out files for each varied parameter
# (Make sure AUTO runs using Python 2)
echo Run run_fold.auto
auto /Users/zhugchzo/Desktop/2paper_code/analyze_equations/study_system/run_fold.auto

echo Run run_sup_hopf.auto
auto /Users/zhugchzo/Desktop/2paper_code/analyze_equations/study_system/run_sup_hopf.auto

echo Run run_trans.auto
auto /Users/zhugchzo/Desktop/2paper_code/analyze_equations/study_system/run_trans.auto

