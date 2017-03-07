# mitgcm_surface_tracer
[![Build Status](https://travis-ci.org/jbusecke/mitgcm_surface_tracer.svg?branch=master)](https://travis-ci.org/jbusecke/mitgcm_surface_tracer)
[![codecov](https://codecov.io/gh/jbusecke/mitgcm_surface_tracer/branch/master/graph/badge.svg)](https://codecov.io/gh/jbusecke/mitgcm_surface_tracer)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)

This package should enable [MITgcm](http://mitgcm.org) users to design,execute and analyze surface tracer stirring experiments (link to pub).
It is heavily under construction under the moment and not useable yet...


## Setting up the conda environment
Using conda as package manager is recommended. Installation instructions can be found **MISSING LINK**
module add anaconda/4.1.1-python-2.7.12

The `tracer_processing_env.yml`
can be used to install all dependencies as a python environment.
```
cd ~TBD~
# Create the environment (this needs to be done only once)
conda env create -f tracer_processing_env.yml

#activate environment
source activate tracer_processing
```
For more information on maintaining conda environments see https://conda.io/docs/using/envs.html



# Tracer Experiment Docs

## Setup

### Installation

Mitgcm

Scripts

Proposed directory structure

### MITgcm 'hacks'

Offline Velocities

Tracer initial Conditions

Tracer Model Scripts

​	Setup Python environment (conda env etc)

​	Structure

​

# Setup of Surface Tracer Experiments

## Configure Mitgcm

TBW

## Prepare Input for MITgcm_Test

- Setup conda evironment with required modules (TBD)
- Convert AVISO velocities to mitgcm binary files (interpolate_aviso_mitgcm.py)
- Convert Tracer fields to mitgcm binary files

## Configure experiments

TBW

## Analize results

- Calculate KOC
- Calculate Keff/TFR -->
