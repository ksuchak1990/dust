#!/bin/bash

# Run the particle filter experiments on the faculty linux computers
# IMPORTANT: set the number of experiments to run below (determined by length of 'param_list'
# variable in StationSim-ARCExperiments.py)

for i in {1..272}
do
 printf " \n\n ********************* \n\n  *** EXPERIMENT $i *** \n\n ********************* \n\n"
 python3 ./run_pf.py $i
done 
