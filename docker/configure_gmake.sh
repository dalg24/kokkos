#!/bin/bash

EXTRA_ARGS=("$@")

ARGS=(
  --prefix=$KOKKOS_PREFIX
  --arch=Volta70,SNB
  --with-serial
  --with-openmp
  --with-cuda
  --with-cuda-options=enable_lambda
)

$KOKKOS_PATH/generate_makefile.bash "${ARGS[@]}" "${EXTRA_ARGS[@]}" 
