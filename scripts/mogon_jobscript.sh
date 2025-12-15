#!/bin/bash

#SBATCH -M mogonki
#SBATCH -J Cuckoo-Filter    # Job name
#SBATCH -o \%x_\%j.out      # Specify stdout output file where \%j expands to jobID and \%x to JobName
#SBATCH -A ki-gpu4science       # Account name
#SBATCH -p a100ai           # Queue name
#SBATCH -n 1                # Number of tasks
#SBATCH -c 16               # Number of CPUs
#SBATCH --gres=gpu:8        # Total number of GPUs
#SBATCH --mem=64G           # Memory per node
#SBATCH -t 360              # Time in minutes

set -e

module purge
module load tools/Meson/1.4.0-GCCcore-13.3.0
module use /apps/easybuild/2025/cuda/modules/all
module load tools/Ninja
module load system/CUDA

srun meson compile -C build

srun ./scripts/run_multi_gpu_scaling.py
