#!/bin/bash
#
#SBATCH --job-name=drl1_500
#SBATCH --output=log.out
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --ntasks-per-core 1
#SBATCH --threads-per-core 1
#SBATCH --time=2-00:00:00
#

module purge

source /scratch-fast/gsagot/miniconda3/etc/profile.d/conda.sh
export PATH="/scratch-fast/gsagot/miniconda3/bin:$PATH"

while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
  conda deactivate
done

echo "Before activating conda environment:"
conda info
conda --version
python3 --version

# Activate your environment
conda activate pbo368
conda info

echo "After activating conda environment:"
conda --version
python3 --version


echo "Which pbo: $(which pbo)"

module load gcc openmpi vtk/latest felicia/latest mtc/tsv eigen/latest hdf5/latest cmake/latest git/latest petsc/latest mtc/latest

pbo airfoil.json

conda deactivate

