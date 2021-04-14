#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#endif

#include <benchmark/benchmark.h>

struct Kokkos_ {};
struct Thrust_ {};
struct SYCL_ {};
struct HIP_ {};
struct CUDA_ {};
struct OMP_ {};

template <class>
struct AXPY;

template <>
struct AXPY<Kokkos_> {
  template <class ExecutionSpace, class View>
  AXPY(ExecutionSpace const &s, View x, View y) {
    run(s, x, y);
  }

  template <class ExecutionSpace, class View>
  void run(ExecutionSpace const &s, View x, View y) {
    typename View::value_type a = 2;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(s, 0, x.size()),
        KOKKOS_LAMBDA(int i) { y[i] = a * x[i] + y[i]; });
    s.fence();
  }
};

#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
template <>
struct AXPY<Thrust_> {
  template <class ExecutionSpace, class View>
  AXPY(ExecutionSpace const &s, View x, View y) {
    run(s, x, y);
  }

  template <class ExecutionSpace, class View>
  void run(ExecutionSpace const &s, View x, View y) {
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
#endif

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
template <>
struct AXPY<OMP_> {
  template <class ExecutionSpace, class View>
  AXPY(ExecutionSpace const &s, View x, View y) {
    using T     = typename View::value_type;
    T const a   = 2;
    int const n = x.size();
    T *xp       = x.data();
    T *yp       = y.data();
#pragma omp target teams distribute parallel for is_device_ptr(xp, yp)
    for (int i = 0; i < n; ++i) {
      yp[i] = a * xp[i] + yp[i];
    }
    s.fence();
  }
};
#endif

#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
template <class T>
__global__ void impl(int n, T a, T *x, T *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

template <>
#ifdef KOKKOS_ENABLE_HIP
struct AXPY<HIP_>
#else
struct AXPY<CUDA_>
#endif
{
  template <class ExecutionSpace, class View>
  AXPY(ExecutionSpace const &s, View x, View y) {
    run(s, x, y);
  }

  template <class ExecutionSpace, class View>
  void run(ExecutionSpace const &s, View x, View y) const {
    typename View::value_type a = 2;
    int n                       = x.size();
    int m                       = 512;
    impl<<<(n + m - 1) / m, m>>>(n, a, x.data(), y.data());
    s.fence();
  }
};
#endif

#if defined(KOKKOS_ENABLE_SYCL)
template <>
struct AXPY<SYCL_>
{
  static cl::sycl::queue sycl_queue;

  template <class ExecutionSpace, class View>
  AXPY(ExecutionSpace const &, View x, View y) {
   // Initialization
   sycl_queue.submit([&](cl::sycl::handler &cgh) {
      auto * x_ = x.data();
      auto * y_ = y.data();
      typename View::value_type a = 2;
      cgh.parallel_for(
        cl::sycl::range<1>(x.size()), [=](cl::sycl::item<1> itemId) {
          const int i = itemId.get_id();
          y_[i] = a * x_[i] + y_[i];
        });
    });
    sycl_queue.wait();
  }
};
auto AXPY<SYCL_>::sycl_queue  = cl::sycl::queue(cl::sycl::gpu_selector());
#endif

template <class>
struct DOT;

template <>
struct DOT<Kokkos_> {
  template <class ExecutionSpace, class View>
  DOT(ExecutionSpace const &s, View x, View y) {
    run(s, x, y);
  }

  template <class ExecutionSpace, class View>
  void run(ExecutionSpace const &s, View x, View y) {
    using T = typename View::value_type;
    T r{};
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>(s, 0, x.size()),
        KOKKOS_LAMBDA(int i, T &s) { s += x[i] * y[i]; }, r);
    s.fence();
  }
};

#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
template <>
struct DOT<Thrust_> {
  template <class ExecutionSpace, class View>
  DOT(ExecutionSpace const &s, View x, View y) {
    using T                       = typename View::value_type;
    int n                         = x.size();
    thrust::device_ptr<T> x_first = thrust::device_pointer_cast<T>(x.data());
    thrust::device_ptr<T> y_first = thrust::device_pointer_cast<T>(y.data());
    auto r = thrust::inner_product(x_first, x_first + n, y_first, 0);
    (void)r;
    s.fence();
  }
};
#endif

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
template <>
struct DOT<OMP_> {
  template <class ExecutionSpace, class View>
  DOT(ExecutionSpace const &s, View x, View y) {
    using T     = typename View::value_type;
    int const n = x.size();
    T *xp       = x.data();
    T *yp       = y.data();

    double result = 0.;
#pragma omp target teams distribute parallel for \
      is_device_ptr(xp,yp) reduction(+:result)
    for (int i = 0; i < n; ++i) {
      result += xp[i] * yp[i];
    }
    s.fence();
  }
};
#endif

#if defined(KOKKOS_ENABLE_SYCL)
template <>
struct DOT<SYCL_> {
  static cl::sycl::queue sycl_queue;

  template <class ExecutionSpace, class View>
  DOT(ExecutionSpace const &, View x, View y) {
      double result = 0.;
      auto result_ptr = static_cast<double*>(
        sycl::malloc(sizeof(result), sycl_queue, sycl::usm::alloc::shared));
      sycl_queue.submit([&](cl::sycl::handler &cgh) {
        auto * x_ = x.data();
        auto * y_ = y.data();  
        auto reduction = cl::sycl::ONEAPI::reduction(result_ptr, std::plus<>());
        cgh.parallel_for(
          cl::sycl::nd_range<1>(x.size(), 128), reduction, [=](cl::sycl::nd_item<1> itemId, auto& sum) {
            const int i = itemId.get_global_id();
            sum.combine(x_[i]*y_[i]);
          });
      });
      sycl_queue.wait();
      sycl_queue.memcpy(&result, result_ptr, sizeof(result));
      sycl_queue.wait();
  }
};
auto DOT<SYCL_>::sycl_queue = cl::sycl::queue(cl::sycl::gpu_selector());
#endif

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
#if defined(KOKKOS_ENABLE_HIP)
  using ExecutionSpace = Kokkos::Experimental::HIP;
#elif defined(KOKKOS_ENABLE_CUDA)
  using ExecutionSpace = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
  using ExecutionSpace = Kokkos::Experimental::OpenMPTarget;
#else
  using ExecutionSpace = Kokkos::Experimental::SYCL;
#endif
  int n = state.range(0);
  ExecutionSpace space{};
  Kokkos::View<T *, ExecutionSpace> x("x", n);
  Kokkos::View<T *, ExecutionSpace> y("y", n);
  K<W>(space, x, y);  // warm-up
  for (auto _ : state) {
    K<W>(space, x, y);
  }
  state.counters["Bandwidth"] =
      benchmark::Counter(factor<K<W>>::value * sizeof(T) * n,
                         benchmark::Counter::kIsIterationInvariantRate);
}
#define REGISTER_BENCHMARK(TAG, KERNEL, TYPE)       \
  BENCHMARK_TEMPLATE(BM_generic, TAG, KERNEL, TYPE) \
      ->RangeMultiplier(8)                          \
      ->Range(1024, 8 << 24)                        \
      ->UseRealTime();

REGISTER_BENCHMARK(Kokkos_, AXPY, int);
REGISTER_BENCHMARK(Kokkos_, AXPY, double);
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
REGISTER_BENCHMARK(Thrust_, AXPY, int);
REGISTER_BENCHMARK(Thrust_, AXPY, double);
#endif
#ifdef KOKKOS_ENABLE_HIP
REGISTER_BENCHMARK(HIP_, AXPY, int);
REGISTER_BENCHMARK(HIP_, AXPY, double);
#endif
#ifdef KOKKOS_ENABLE_CUDA
REGISTER_BENCHMARK(CUDA_, AXPY, int);
REGISTER_BENCHMARK(CUDA_, AXPY, double);
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
REGISTER_BENCHMARK(OMP_, AXPY, int);
REGISTER_BENCHMARK(OMP_, AXPY, double);
#endif
#ifdef KOKKOS_ENABLE_SYCL
REGISTER_BENCHMARK(SYCL_, AXPY, int);
REGISTER_BENCHMARK(SYCL_, AXPY, double);
#endif
REGISTER_BENCHMARK(Kokkos_, DOT, int);
REGISTER_BENCHMARK(Kokkos_, DOT, double);
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
REGISTER_BENCHMARK(Thrust_, DOT, int);
REGISTER_BENCHMARK(Thrust_, DOT, double);
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
REGISTER_BENCHMARK(OMP_, DOT, int);
REGISTER_BENCHMARK(OMP_, DOT, double);
#endif
#ifdef KOKKOS_ENABLE_SYCL
REGISTER_BENCHMARK(SYCL_, DOT, int);
REGISTER_BENCHMARK(SYCL_, DOT, double);
#endif

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  Kokkos::finalize();
}
