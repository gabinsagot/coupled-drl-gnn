#!/bin/bash
#
#SBATCH --job-name=dataset
#SBATCH --output=log.out
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --ntasks-per-core 1
#SBATCH --threads-per-core 1
#SBATCH --time=168:00:00
#

module load cimlibxx/master

# command:
python -m main \
        --dataset_config_file=./config/morphed_airfoil.json \
        --create_configs_pool=True \
