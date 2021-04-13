#include <Kokkos_Core.hpp>

#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>

#include <benchmark/benchmark.h>

struct Kokkos_ {};
struct Thrust_ {};
struct HIP_ {};
struct CUDA_ {};

template <class>
struct AXPY;

template <>
struct AXPY<Kokkos_> {
  template <class ExecutionSpace, class View>
  AXPY(ExecutionSpace const &s, View x, View y) {
    typename View::value_type a = 2;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, x.size()),
        KOKKOS_LAMBDA(int i) { y[i] = a * x[i] + y[i]; });
    s.fence();
  }
};

template <>
struct AXPY<Thrust_> {
  template <class ExecutionSpace, class View>
  AXPY(ExecutionSpace const &s, View x, View y) {
    using T                       = typename View::value_type;
    T a                           = 2;
    int n                         = x.size();
    thrust::device_ptr<T> x_first = thrust::device_pointer_cast<T>(x.data());
    thrust::device_ptr<T> y_first = thrust::device_pointer_cast<T>(y.data());
    thrust::transform(
        x_first, x_first + n, y_first, y_first,
        [=] __host__ __device__(T x_, T y_) { return a * x_ + y_; });
    s.fence();
  }
};

#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
template <>
#ifdef KOKKOS_ENABLE_HIP
struct AXPY<HIP_>
#else
struct AXPY<CUDA_>
#endif
{
  template <class ExecutionSpace, class View>
  AXPY(ExecutionSpace const &s, View x, View y) {
#ifdef KOKKOS_ENABLE_CUDA
    run(s, x, y);
  }

  template <class ExecutionSpace, class View>
  void run(ExecutionSpace const &s, View x, View y) const {
#endif
    typename View::value_type a = 2;
    int n                       = x.size();
    int m                       = 512;
    impl<<<(n + m - 1) / m, m>>>(n, a, x.data(), y.data());
    s.fence();
  }
  template <class T>
  __global__ static void impl(int n, T a, T *x, T *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
      y[i] = a * x[i] + y[i];
    }
  }
};
#endif

template <class>
struct DOT;

template <>
struct DOT<Kokkos_> {
  template <class ExecutionSpace, class View>
  DOT(ExecutionSpace const &s, View x, View y) {
    using T = typename View::value_type;
    T r{};
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>(0, x.size()),
        KOKKOS_LAMBDA(int i, T &s) { s += x[i] * y[i]; }, r);
  }
};

template <>
struct DOT<Thrust_> {
  template <class ExecutionSpace, class View>
  DOT(ExecutionSpace const &s, View x, View y) {
    using T                       = typename View::value_type;
    T a                           = 2;
    int n                         = x.size();
    thrust::device_ptr<T> x_first = thrust::device_pointer_cast<T>(x.data());
    thrust::device_ptr<T> y_first = thrust::device_pointer_cast<T>(y.data());
    auto r = thrust::inner_product(x_first, x_first + n, y_first, 0);
    s.fence();
  }
};

template <class>
struct factor;

template <class Tag>
struct factor<AXPY<Tag>> {
  static constexpr int value = 3;
};

template <class Tag>
struct factor<DOT<Tag>> {
  static constexpr int value = 2;
};

template <class W, template <class> class K, class T>
void BM_generic(benchmark::State &state) {
#ifdef KOKKOS_ENABLE_HIP
  using ExecutionSpace = Kokkos::Experimental::HIP;
#else
  using ExecutionSpace = Kokkos::Cuda;
#endif
  int n = state.range(0);
  ExecutionSpace space{};
  Kokkos::View<T *, ExecutionSpace> x("x", n);
  Kokkos::View<T *, ExecutionSpace> y("y", n);
  K<W>(space, x, y);  // warm-up
  double c = 0;
  for (auto _ : state) {
    K<W>(space, x, y);
    ++c;
  }
  state.counters["Bandwidth"] = benchmark::Counter(
      c * factor<K<W>>::value * sizeof(T) * n, benchmark::Counter::kIsRate);
}
#define REGISTER_BENCHMARK(TAG, KERNEL, TYPE)       \
  BENCHMARK_TEMPLATE(BM_generic, TAG, KERNEL, TYPE) \
      ->RangeMultiplier(8)                          \
      ->Range(1024, 8 << 24)

REGISTER_BENCHMARK(Kokkos_, AXPY, int);
REGISTER_BENCHMARK(Kokkos_, AXPY, double);
REGISTER_BENCHMARK(Thrust_, AXPY, int);
REGISTER_BENCHMARK(Thrust_, AXPY, double);
#ifdef KOKKOS_ENABLE_HIP
REGISTER_BENCHMARK(HIP_, AXPY, int);
REGISTER_BENCHMARK(HIP_, AXPY, double);
#endif
#ifdef KOKKOS_ENABLE_CUDA
REGISTER_BENCHMARK(CUDA_, AXPY, int);
REGISTER_BENCHMARK(CUDA_, AXPY, double);
#endif
REGISTER_BENCHMARK(Kokkos_, DOT, int);
REGISTER_BENCHMARK(Kokkos_, DOT, double);
REGISTER_BENCHMARK(Thrust_, DOT, int);
REGISTER_BENCHMARK(Thrust_, DOT, double);

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  Kokkos::finalize();
}