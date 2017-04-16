#!/bin/bash

#SBATCH --account=ocp
#SBATCH --exclusive
#SBATCH -N 2
#SBATCH -J aviso_video
#SBATCH --time=8:00:00
#SBATCH --mail-user=julius@ldeo.columbia.edu
#SBATCH --mail-type=ALL

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
source activate standard
python -c 'from xarrayutils.visualization import mitgcm_Movie; mitgcm_Movie("'$SLURM_SUBMIT_DIR'"); exit();'
