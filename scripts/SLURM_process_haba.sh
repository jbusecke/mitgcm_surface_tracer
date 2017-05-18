#!/bin/bash

#SBATCH --account=ocp
#SBATCH --exclusive
#SBATCH -N 4
#SBATCH -J tr_proc
#SBATCH --time=3:00:00
#SBATCH --mail-user=julius@ldeo.columbia.edu
#SBATCH --mail-type=ALL

RUNDIR=$SLURM_SUBMIT_DIR
VALIDPATH="/rigel/ocp/users/jb3210/aviso_surface_tracer/offline_velocities/aviso_DUACS2014_daily_msla/interpolated/validmask_combined.bin"
source activate standard

ODIR="$RUNDIR/output"
# reset months
RESET=3

cd $RUNDIR
echo 'REMOVING OLD DIRECTORIES'

# current folder structure
rm -r output
rm -r plots
rm -r movie*
rm -r python_*

# remake the directory
mkdir $ODIR

echo 'START PYTHON READOUT'
echo 'Test new version'
python -c 'from mitgcm_surface_tracer.tracer_processing import main;\
main("'$RUNDIR/'","'$ODIR/'","'$VALIDPATH'",\
    koc_interval=10,\
    kappa=63,\
    raw_output=True,\
    spin_up_time = float("'$RESET'"));\
exit()' > pyout.txt


echo "PBS PROCESSING SCRIPT DONE"
sbatch SLURM_plot.sh
