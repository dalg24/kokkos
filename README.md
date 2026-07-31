# Kokkos Ecosystem Build Instructions

This repository contains a meta-build system for the Kokkos ecosystem, designed to simplify the build and installation process for reviewers.

## Overview

This build includes the following Kokkos components:
- **Kokkos MDSPAN** (version 5d4eb20)
- **Kokkos Core** (version 5.2.0)
- **Kokkos Kernels** (version 5.2.0)
- **Kokkos Tools** (version afced73)
- **Kokkos FFT** (version 2.0.0) - Optional, requires FFTW

## Prerequisites

- CMake 3.28 or later
- C++ compiler with C++20 support
- (Optional) FFTW library for FFT support
- (Optional) CUDA 12.x (Kokkos itself supports 13.x too, but some of the benchmarks don't)
- (Optional) ROCM 6.4 or newer

## Build Instructions

### 1. Configure the Build

```bash
cmake -B builddir
```

To enable FFT support:
```bash
cmake -B builddir -DENABLE_FFT=ON
```

The above only enables the Serial backend in Kokkos. To build with CUDA or HIP use:
```bash
cmake -B builddir -DENABLE_FFT=ON -DKokkos_ENABLE_CUDA=ON
```
or
```bash
cmake -B builddir -DENABLE_FFT=ON -DKokkos_ENABLE_HIP=ON
```

Note: this requires to build on a machine with a respective GPU, and the CUDA or ROCM toolchain available in the environment.

Note: while KokkosKernels itself works with CUDA 13, the KokkosKernels benchmarks are not yet compatible with it. CUDA 13
had some breaking changes which are not accounted for in the bundled Kokkos 5.2 version. These changes affect device
properties such as clock rates, which are reported in the benchmarks.


### 2. Build the Ecosystem

```bash
cmake --build builddir
```

### 3. Install

```bash
cmake --install builddir --prefix installdir
```

### 4. Set Environment Variable

```bash
export KOKKOS_ECOSYSTEM_ROOT=$PWD/installdir
```

## Run a KokkosKernels Benchmark with a Kokkos Tool

Linux:
```bash
export KOKKOS_TOOLS_LIBS=${KOKKOS_ECOSYSTEM_ROOT}/lib64/libkp_space_time_stack.so
builddir/_deps/kokkoskernels-build/benchmarks/sparse/KokkosKernels_sparse_spmv_benchmark
```

MacOS:
```bash
export KOKKOS_TOOLS_LIBS=${KOKKOS_ECOSYSTEM_ROOT}/lib/libkp_space_time_stack.dylib
builddir/_deps/kokkoskernels-build/benchmarks/sparse/KokkosKernels_sparse_spmv_benchmark
```

## Testing with Kokkos Tutorials

### 1. Verify and Extract Tutorials

```bash
grep tutorials kokkos-ecosystem-SHA-256.txt | shasum -c
tar xzf kokkos-tutorials-6dd51fb.tar.gz
cd kokkos-kokkos-tutorials-6dd51fb/
```

### 2. Build a Tutorial Exercise

```bash
cd Exercises/01/Solution
cmake -B builddir -DKokkos_ROOT=$KOKKOS_ECOSYSTEM_ROOT
cmake --build builddir
```

### 3. Run with Kokkos Tools

**Linux:**
```bash
./builddir/01_Exercise --kokkos-tools-libs=$KOKKOS_ECOSYSTEM_ROOT/lib/libkp_space_time_stack.so
```

**macOS:**
```bash
./builddir/01_Exercise --kokkos-tools-libs=$KOKKOS_ECOSYSTEM_ROOT/lib/libkp_space_time_stack.dylib
```

## Directory Structure

```
.
├── CMakeLists.txt
├── kokkos-mdspan-5d4eb20.tar.gz
├── kokkos-core-5.2.0.tar.gz
├── kokkos-kernels-5.2.0.tar.gz
├── kokkos-fft-2.0.0.tar.gz
├── kokkos-tools-afced73.tar.gz
├── kokkos-tutorials-6dd51fb.tar.gz
├── kokkos-ecosystem-SHA-256.txt
├── builddir/              (created during build)
└── installdir/            (created during install)
```

## Troubleshooting

## Additional Information

For more information about Kokkos, visit:
- Kokkos Core: https://kokkos.org/kokkos-core-wiki/get-started.html
