#!/bin/bash

#SBATCH --job-name=mpi_test
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --account=tc060-ruairiph

export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1

# Define some paths
export WORK=/work/tc060/tc060/ruairiph_tc060
export JULIA="$WORK/julia-1.11.1/bin/julia"  # The julia executable
export PATH="$PATH:$WORK/julia-1.11.1/bin"  # The folder of the julia executable
export JULIA_DEPOT_PATH="$WORK/.julia"

# Load correct mpi module
module load openmpi/4.1.6-cuda-12.4

# export OMPI_MCA_btl=vader,self
export OMPI_MCA_btl=self

# scontrol show job $SLURM_JOBID

# Run program
srun --ntasks=4 --tasks-per-node=4 --hint=nomultithread julia --project=$WORK/MPIGPUVicsek ./test_script.jl
