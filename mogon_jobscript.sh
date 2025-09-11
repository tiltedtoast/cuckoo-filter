#!/bin/bash

#SBATCH -M mogonki
#SBATCH -J Cuckoo-Filter    # Job name
#SBATCH -o \%x_\%j.out      # Specify stdout output file where \%j expands to jobID and \%x to JobName
#SBATCH -A ki-acccomp       # Account name
#SBATCH -p a100ai           # Queue name
#SBATCH -n 1                # Number of tasks
#SBATCH -c 8                # Number of CPUs
#SBATCH --gres=gpu:1        # Total number of GPUs
#SBATCH --mem=64G           # Memory per node
#SBATCH -t 360              # Time in minutes

set -e

module purge
module load tools/Meson/1.4.0-GCCcore-13.3.0
module use /apps/easybuild/2024/cuda/modules/all
module load tools/Ninja
module load system/CUDA

srun meson setup build
srun ninja -C build

srun ./build/benchmark_bucket_size