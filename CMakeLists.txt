cmake_minimum_required(VERSION 3.28)
project(KokkosEcosystem)

include(FetchContent)

FetchContent_Declare(
  KokkosMDSPAN
  URL "${CMAKE_CURRENT_SOURCE_DIR}/kokkos-mdspan-5d4eb20.tar.gz"
  URL_HASH SHA256=8fee53a8f3673dd0d90fc0ef2ec564633fabbf539e525022106e13cd5b70b1b7
)
FetchContent_MakeAvailable(KokkosMDSPAN)

FetchContent_Declare(
  KokkosCore
  URL "${CMAKE_CURRENT_SOURCE_DIR}/kokkos-core-5.2.0.tar.gz"
  URL_HASH SHA256=54993e0682d80b78939bbf260490f8cf31428bb883c0309961369997f15d94df
)
FetchContent_MakeAvailable(KokkosCore)

FetchContent_Declare(
  KokkosKernels
  URL "${CMAKE_CURRENT_SOURCE_DIR}/kokkos-kernels-5.2.0.tar.gz"
  URL_HASH SHA256=57f278e7a25940f9a1237d8b370a1de7f09ea282626b4594e4880042181acdcd
)
FetchContent_MakeAvailable(KokkosKernels)

# requires FFTW
if(ENABLE_FFT)
  FetchContent_Declare(
    KokkosFFT
    URL "${CMAKE_CURRENT_SOURCE_DIR}/kokkos-fft-2.0.0.tar.gz"
    URL_HASH SHA256=f2c1f7b848e68aa214f2dea7820c22e729564167a4d4010170b01c81ad3a0714
  )
  FetchContent_MakeAvailable(KokkosFFT)
endif()

set(BUILD_SHARED_LIBS ON)
FetchContent_Declare(
  KokkosTools
  URL "${CMAKE_CURRENT_SOURCE_DIR}/kokkos-tools-afced73.tar.gz"
  URL_HASH SHA256=2e2bcd8de8fe9d34a42243ff8d5dfdac1f5c42355d0d6b266dab7b352ae7cfa0
)
FetchContent_MakeAvailable(KokkosTools)
