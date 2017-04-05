#!/bin/bash
GRIDDIR='/data/scratch/julius/KOCmaps/offline_velocities/grid_setup'
# OUTDIR='/data/scratch/julius/KOCmaps/offline_velocities/interpolated'
OUTDIR='/swot/SUM05/julius/offline_velocities/interpolated'
DTDIR='/data/scratch/julius/KOCmaps/offline_velocities/raw_dt'
DTFID='dt_global_allsat_msla_uv_'
NRTDIR='/data/scratch/julius/KOCmaps/offline_velocities/raw_nrt'
NRTID='nrt_global_allsat_msla_uv_'

# Specify the year when dt and nrt meet
crossover=2016
endyear=2016
startyear=1993
# startyear=2015

# activate environsment
source activate standard

rm $OUTDIR/*

for y in `seq $startyear $endyear`
do
  echo "Processing $y"
  mkdir $OUTDIR/$y

  if [ $crossover -eq $y ]
  then
    echo "Crossover year"
    python -c 'from mitgcm_surface_tracer.velocity_processing \
    import process_aviso; \
    process_aviso("'$OUTDIR/$y'", "'$GRIDDIR'", "'$DTDIR/$y'", ddir_nrt="'$NRTDIR/$y'");\
    exit();'
  else
    echo "non crossover year"
    python -c 'from mitgcm_surface_tracer.velocity_processing \
    import process_aviso; \
    process_aviso("'$OUTDIR/$y'", "'$GRIDDIR'", "'$DTDIR/$y'");\
    exit();'
  fi
done

# Combine validmask
python -c 'from mitgcm_surface_tracer.velocity_processing \
import combine_validmask; combine_validmask("'$OUTDIR'", shape=(1600,3600));\
exit();'

# softlink into combined directory with consequtive numbers
mkdir $OUTDIR/combined
cd $OUTDIR
a=0
find . -name 'uvel*' | while read line; do
  new=$(printf "%010d" "$a") #04 pad to length of 4
  ln -s "$line" "$OUTDIR/combined/uvel.$new.data"
  let a=a+1
done
a=0
find . -name 'vvel*' | while read line; do
  new=$(printf "%010d" "$a") #04 pad to length of 4
  ln -s "$line" "$OUTDIR/combined/vvel.$new.data"
  let a=a+1
done
