#!/bin/sh

# for building locally
# CUDA_PATH=/usr	nvcc located at /usr/bin/nvcc
# TARGET_ARCH=x86_64	i7-2600
# SMS=30		GT7440, compute capability=3.0
CUDA_PATH=/usr TARGET_ARCH=x86_64 SMS=30 make
