#!/usr/bin/env bash

module purge
module load tools/Meson/1.4.0-GCCcore-13.3.0
module use /apps/easybuild/2025/cuda/modules/all
module load tools/Ninja
module load system/CUDA