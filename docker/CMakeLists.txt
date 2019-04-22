cmake_minimum_required(VERSION 3.10)

find_package(Kokkos REQUIRED)

add_executable(bytes_and_flops.exe main.cpp)
target_link_libraries(bytes_and_flops.exe Kokkos::Kokkos)
target_include_directories(bytes_and_flops.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


#find_package(PkgConfig)
#pkg_check_modules(PC_Kokkos REQUIRED IMPORTED_TARGET kokkos)
#target_link_libraries(bytes_and_flops.exe PkgConfig::PC_Kokkos)
#target_link_libraries(bytes_and_flops.exe ${PC_Kokkos_LDFLAGS})

#target_link_libraries(bytes_and_flops.exe kokkos)
##                                         ^^^^^^
##                                         Lowercase (k)okkos
#target_include_directories(bytes_and_flops.exe PRIVATE ${Kokkos_INCLUDE_DIRS})
