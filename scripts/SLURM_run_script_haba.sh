#!/bin/bash

#SBATCH --account=ocp
#SBATCH --exclusive
#SBATCH -N 6
#SBATCH -J aviso_surface_tracer
#SBATCH --time=60:00:00
#SBATCH --mail-user=julius@ldeo.columbia.edu
#SBATCH --mail-type=ALL

echo "Running MITgcm in $RUNDIR"

NPROC=128

module load intel-parallel-studio/2017 netcdf-fortran/4.4.4 netcdf/gcc/64/4.4.0
module list

cd $SLURM_SUBMIT_DIR

rm *.meta
rm *.data
rm STD*
rm slurm*

# write the tracer source path into file
ls -l init_tracer.bin > tracersource.txt

mpirun -n $NPROC ./mitgcmuv

sbatch SLURM_process_haba.sh
