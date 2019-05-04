#!/bin/bash

EXTRA_ARGS=("$@")

ARGS=(
  -D CMAKE_INSTALL_PREFIX=$KOKKOS_PREFIX
  -D CMAKE_BUILD_TYPE=Release
  -D CMAKE_CXX_COMPILER=$KOKKOS_PATH/bin/nvcc_wrapper
  -D KOKKOS_ARCH="Volta70;SNB"
  -D KOKKOS_ENABLE_SERIAL=ON
  -D KOKKOS_ENABLE_OPENMP=ON
  -D KOKKOS_ENABLE_CUDA=ON
  -D KOKKOS_ENABLE_CUDA_LAMBDA=ON
  -D CMAKE_CXX_COMPILER=clang++
)

cmake "${ARGS[@]}" "${EXTRA_ARGS[@]}" $KOKKOS_PATH
