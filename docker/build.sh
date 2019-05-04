#!/bin/bash

EXTRA_ARGS=("$@")

ARGS=(
  -D CMAKE_CXX_COMPILER=nvcc_wrapper
  -D CMAKE_FIND_DEBUG_MODE=ON
#  -D CMAKE_PREFIX_PATH=/opt/kokkos
  -D CMAKE_MODULE_PATH=/scratch/source/cmake
  -D DETECT="FindKokkos"
  -D CMAKE_CXX_COMPILER=clang++
  -D CMAKE_PREFIX_PATH="/opt/kokkos;/opt/kokkos/lib/CMake"
)

rm -rf CMake*

cmake "${ARGS[@]}" "${EXTRA_ARGS[@]}" /scratch/source

make VERBOSE=1
