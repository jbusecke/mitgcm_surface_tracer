#!/bin/bash

#SBATCH --account=ocp
#SBATCH --exclusive
#SBATCH -N 6
#SBATCH -J tr_run_hf
#SBATCH --time=18:00:00
#SBATCH --mail-user=julius@ldeo.columbia.edu
#SBATCH --mail-type=ALL

NPROC=128

module load intel-parallel-studio/2017 netcdf-fortran/4.4.4 netcdf/gcc/64/4.4.0
module list

cd $SLURM_SUBMIT_DIR

rm *.meta
rm *.data
rm STD*

mpirun -n $NPROC ./mitgcmuv

sbatch SLURM_video_aviso_surface_tracer_habanero.sh
