#!/bin/bash
#
#SBATCH --job-name=channelflow
#SBATCH --output=log.out
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --ntasks-per-core 1
#SBATCH --threads-per-core 1
#SBATCH --time=7-00:00:00
#
module load gcc openmpi vtk/latest felicia/latest mtc/tsv eigen/latest hdf5/latest cmake/latest git/latest petsc/latest mtc/latest
module load cimlibxx/master

mpirun /scratch-fast/tmichel/cimlib/cimlibxx/build/CFD_windpv/bin/cimlib_CFD_driver lanceur/Principale.mtc
