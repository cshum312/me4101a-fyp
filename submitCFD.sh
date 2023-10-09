#!/bin/bash

# queue
#PBS -q normal
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=16GB
#PBS -l walltime=00:30:00
#PBS -P Personal
#PBS -N channel_flow
#PBS -j oe

# commands for parallel pimpleFoam after decompose
cd $PBS_O_WORKDIR
module load openfoam/v1912
fluentMeshToFoam newmesh.msh
checkMesh
changeDictionary
decomposePar
mpirun -np 4 pimpleFoam -parallel > log_test
mpirun -np 4 pimpleFoam -parallel -postProcess
reconstructPar

