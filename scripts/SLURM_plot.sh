#!/bin/bash

#SBATCH --account=ocp
#SBATCH --exclusive
#SBATCH -N 2
#SBATCH -J python_plot
#SBATCH --time=2:00:00
#SBATCH --mail-user=julius@ldeo.columbia.edu
#SBATCH --mail-type=ALL

if [ "$1" == "dev" ]
then
  echo "DEV MODE ACTIVE"
  RUNDIR="/Volumes/EXTERNAL_WORK/run_KOC_daily_LAT"
  # PYTHONDIR="/Users/juliusbusecke/Work/PROJECTS/COLL_RYAN/ROUTINES/SETUP"
  # PBSDIR=$PYTHONDIR
  VALIDPATH="/Users/juliusbusecke/Work/PROJECTS/COLL_RYAN/OUTPUT/AVISO_validmask/validmask_combined.bin"
else
  RUNDIR=$PBS_O_WORKDIR
  # PYTHONDIR=/vega/physo/users/jb3210/tracer_model_scripts
  # PBSDIR=$RUNDIR
  VALIDPATH="/vega/physo/users/jb3210/offline_velocities/aviso_DUACS2014_daily_msla/interpolated/validmask_combined.bin"
  source activate standard
fi


PDIR="$RUNDIR/plots"
rm -r plots

# remake the directory
mkdir $PDIR

echo 'START PYTHON QC PLOTS'
python -c 'from mitgcm_surface_tracer.tracer_visualization import main;\
main("'$RUNDIR/'","'$PDIR/'");\
exit();'
