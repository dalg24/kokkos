find_package(benchmark REQUIRED CONFIG)
add_executable(KokkosBenchmark_Latency latency_bench.cpp)
target_link_libraries(KokkosBenchmark_Latency Kokkos::kokkos benchmark::benchmark)
