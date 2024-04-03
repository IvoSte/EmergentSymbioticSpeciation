#!/bin/bash

# This script runs several experiments with different parameters. Add your 

python experiment_runner.py -m toxin -e _your_toxin_experiment_ -mp -cpu 4
# python experiment_runner.py -m predator_prey -e _your_predator_prey_experiment_ -mp -cpu 4
# python experiment_runner.py -m function_optimization -e _your_function_optimization_experiment_ -mp -cpu 4
# etc...