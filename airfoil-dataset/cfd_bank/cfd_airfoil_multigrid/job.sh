#!/bin/bash
#
#SBATCH --job-name=channel
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

mpirun /scratch-fast/tmichel/cimlib/cimlibxx/build/CFD_windpv/bin/cimlib_CFD_driver lanceur/PrincipaleMeshing.mtc
mv meshes/domain.t meshes/domain_gmsh.t
cp -r OutputMesh/Mesh_00002.t meshes/domain.t
mpirun /scratch-fast/tmichel/cimlib/cimlibxx/build/CFD_windpv/bin/cimlib_CFD_driver lanceur/Principale.mtc
vtu2h5 Resultats/2d/*.vtu --outfile Resultats/simu_fine.xdmf --delete
vtu2h5 Resultats/2dCoarse/*.vtu --outfile Resultats/simu_coarse.xdmf --delete
rm -rf Resultats/2d/ Resultats/2dCoarse/
