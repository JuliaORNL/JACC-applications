 #!/bin/bash

 PROJ_DIR=/gpfs/wolf2/olcf/trn047/scratch/$USER
 export JULIA_DEPOT_PATH=$PROJ_DIR/.julia

 module purge

 ROCM_VERSION=7.2.0

 module load PrgEnv-cray/8.6.0
 module load cray-mpich
 module load rocm/$ROCM_VERSION
 module load julia/1.13.1

 export ROCM_PATH=/opt/rocm-$ROCM_VERSION

 julia --project -e 'using Pkg; Pkg.instantiate()'
