#include <Kokkos_Core.hpp>

#include <benchmark/benchmark.h>

struct Kokkos_ {};
struct CUDA_ {};
struct HIP_ {};

template <class>
struct Latency;

template <>
struct Latency<Kokkos_> {
  template <class ExecutionSpace, class View>
  Latency(ExecutionSpace const &s, View x, int batch_size, int work_size) {
    run(s, x, batch_size, work_size);
  }

  template <class ExecutionSpace, class View>
  void run(ExecutionSpace const &s, View x, int batch_size, int work_size) {
    typename View::value_type a = 2;
    for (int k = 0; k < batch_size; ++k) {
      Kokkos::parallel_for(
          Kokkos::RangePolicy<ExecutionSpace>(s, 0, x.size()),
          KOKKOS_LAMBDA(int i) {
            for (int j = 0; j < work_size; ++j) {
              x[i] += a;
            }
          });
    }
    s.fence();
  }
};

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
template <class T>
__global__ void impl(int n, int work_size, T a, T *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    for (int j = 0; j < work_size; ++j) {
      x[i] += a;
    }
  }
}

template <>
#ifdef KOKKOS_ENABLE_CUDA
struct Latency<CUDA_>
#else
struct Latency<HIP_>
#endif
{
  template <class ExecutionSpace, class View>
  Latency(ExecutionSpace const &s, View x, int batch_size, int work_size) {
    run(s, x, batch_size, work_size);
  }

  template <class ExecutionSpace, class View>
  void run(ExecutionSpace const &s, View x, int batch_size, int work_size) {
    typename View::value_type a = 2;
    int n                       = x.size();
    int m                       = 512;
    for (int k = 0; k < batch_size; ++k) {
      impl<<<(n + m - 1) / m, m>>>(n, work_size, a, x.data());
    }
    s.fence();
  }
};
#endif

template <class W, template <class> class K>
void BM_generic(benchmark::State &state) {
#if defined(KOKKOS_ENABLE_HIP)
  using ExecutionSpace = Kokkos::Experimental::HIP;
#elif defined(KOKKOS_ENABLE_CUDA)
  using ExecutionSpace = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
  using ExecutionSpace = Kokkos::Experimental::OpenMPTarget;
#else
  using ExecutionSpace = Kokkos::Experimental::SYCL;
#endif
  int n          = state.range(0);
  int batch_size = state.range(1);
  int work_size  = state.range(2);
  ExecutionSpace space{};
  Kokkos::View<double *, ExecutionSpace> x("x", n);
  K<W>(space, x, batch_size, work_size);  // warm-up
  int n_iter = 0;
  for (auto _ : state) {
    K<W>(space, x, batch_size, work_size);
    ++n_iter;
  }

  state.counters["Time_per_kernel"] =
      benchmark::Counter(batch_size * n_iter, benchmark::Counter::kIsRate |
                                                  benchmark::Counter::kInvert);
}

#define REGISTER_BENCHMARK(TAG, KERNEL)                     \
  BENCHMARK_TEMPLATE(BM_generic, TAG, KERNEL)               \
      ->Ranges({{1024, 8 << 16}, {1, 2 << 8}, {0, 2 << 4}}) \
      ->UseRealTime();

REGISTER_BENCHMARK(Kokkos_, Latency);
#ifdef KOKKOS_ENABLE_CUDA
REGISTER_BENCHMARK(CUDA_, Latency);
#endif
#ifdef KOKKOS_ENABLE_HIP
REGISTER_BENCHMARK(HIP_, Latency);
#endif

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  Kokkos::finalize();
}
