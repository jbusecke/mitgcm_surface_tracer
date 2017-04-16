#!/bin/bash

#SBATCH --account=ocp
#SBATCH --exclusive
#SBATCH -N 2
#SBATCH -J python_plot
#SBATCH --time=2:00:00
#SBATCH --mail-user=julius@ldeo.columbia.edu
#SBATCH --mail-type=ALL

RUNDIR=$SLURM_SUBMIT_DIR
VALIDPATH="/rigel/ocp/users/jb3210/aviso_surface_tracer/offline_velocities/aviso_DUACS2014_daily_msla/interpolated/validmask_combined.bin"
PDIR="$RUNDIR/plots"

rm -r plots
# remake the directory
mkdir $PDIR


echo 'START PYTHON QC PLOTS'
source activate standard
python -c 'from mitgcm_surface_tracer.tracer_visualization import main;\
main("'$RUNDIR/'","'$PDIR/'");\
exit();'
