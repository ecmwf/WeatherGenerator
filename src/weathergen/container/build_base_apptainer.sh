#!/bin/bash

module purge
module load Stages
module try-load GCC Apptainer-Tools

TMP=$(mktemp -p /p/scratch/atmlaml/apptainer -d) # creates a temporary directory in shared memory
export APPTAINER_TMPDIR=$TMP

apptainer build --fakeroot /p/scratch/atmlaml/apptainer/images/pytorch-24.12-py3-arm64.sif container/pytorch-24.12-py3.def

rm -rf $TMP # this is important to clean up the temporary directory

