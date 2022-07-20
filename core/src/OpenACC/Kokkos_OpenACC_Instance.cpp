/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>

#include <Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_Instance.hpp>
#include <impl/Kokkos_ExecSpaceManager.hpp>

#include <vector>
#include <ostream>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {
uint32_t OpenACCInternal::impl_get_instance_id() const noexcept {
  return m_instance_id;
}

void OpenACCInternal::fence(const std::string& name,
                            openacc_fence_is_static is_static) {
  if (is_static == openacc_fence_is_static::no) {
    Kokkos::Tools::Experimental::Impl::profile_fence_event<
        Kokkos::Experimental::OpenACC>(
        name,
        Kokkos::Tools::Experimental::Impl::DirectFenceIDHandle{
            impl_get_instance_id()},
        [&]() {
          //[DEBUG] disabled due to synchronous behaviors of the current
          // parallel construct implementations. acc_wait_all();
        });
  } else {
    Kokkos::Tools::Experimental::Impl::profile_fence_event<
        Kokkos::Experimental::OpenACC>(
        name,
        Kokkos::Tools::Experimental::SpecialSynchronizationCases::
            GlobalDeviceSynchronization,
        [&]() {  
          acc_wait_all();
        });
  }
}
int OpenACCInternal::concurrency() { return 128000; }  // FIXME_OPENACC
const char* OpenACCInternal::name() { return "OpenACC"; }
void OpenACCInternal::print_configuration(std::ostream& os, bool) {
  // FIXME_OPENACC
  os << "Using OpenACC\n";
}

void* OpenACCInternal::m_scratch_ptr         = nullptr;
int64_t OpenACCInternal::m_scratch_size      = 0;
int* OpenACCInternal::m_lock_array           = nullptr;
uint64_t OpenACCInternal::m_lock_size        = 0;
uint32_t* OpenACCInternal::m_uniquetoken_ptr = nullptr;

void OpenACCInternal::impl_finalize() {
  m_is_initialized = false;
  if (m_lock_array != nullptr) clear_lock_array();

  if (m_uniquetoken_ptr != nullptr)
    Kokkos::kokkos_free<Kokkos::Experimental::OpenACCSpace>(
        m_uniquetoken_ptr);
}
void OpenACCInternal::impl_initialize() { m_is_initialized = true; }
int OpenACCInternal::impl_is_initialized() { return m_is_initialized ? 1 : 0; }

OpenACCInternal* OpenACCInternal::impl_singleton() {
  static OpenACCInternal self;
  return &self;
}

void OpenACCInternal::verify_is_process(const char* const label) {
  if (acc_on_device(acc_device_not_host)) {
    std::string msg(label);
    msg.append(" ERROR: in parallel");
    Kokkos::Impl::throw_runtime_exception(msg);
  }
}

void OpenACCInternal::verify_initialized(const char* const label) {
  if (0 == Kokkos::Experimental::OpenACC().impl_is_initialized()) {
    std::string msg(label);
    msg.append(" ERROR: not initialized");
    Kokkos::Impl::throw_runtime_exception(msg);
  }
}


void OpenACCInternal::clear_scratch() {
  Kokkos::Experimental::OpenACCSpace space;
  space.deallocate(m_scratch_ptr, m_scratch_size);
  m_scratch_ptr  = nullptr;
  m_scratch_size = 0;
}

void OpenACCInternal::clear_lock_array() {
  if (m_lock_array != nullptr) {
    Kokkos::Experimental::OpenACCSpace space;
    space.deallocate(m_lock_array, m_lock_size);
    m_lock_array = nullptr;
    m_lock_size  = 0;
  }
}

void* OpenACCInternal::get_scratch_ptr() { return m_scratch_ptr; }

int* OpenACCInternal::get_lock_array(int num_teams) {
  // FIXME_OPENACC - Need to be updated.
  Kokkos::Experimental::OpenACCSpace space;
  int max_active_league_size = MAX_ACTIVE_THREADS / 32;
  int lock_array_elem =
      (num_teams > max_active_league_size) ? num_teams : max_active_league_size;
  if (m_lock_size < (lock_array_elem * sizeof(int))) {
    space.deallocate(m_lock_array, m_lock_size);
    m_lock_size  = lock_array_elem * sizeof(int);
    m_lock_array = static_cast<int*>(space.allocate(m_lock_size));

    int* h_lock_array = static_cast<int*>(malloc(m_lock_size));

    for (int i = 0; i < lock_array_elem; ++i) h_lock_array[i] = 0;

    acc_memcpy_to_device(m_lock_array, h_lock_array, m_lock_size);

    free(h_lock_array);
  }

  return m_lock_array;
}



}  // Namespace Impl

OpenACC::OpenACC()
    : m_space_instance(Impl::OpenACCInternal::impl_singleton()) {}

const char* OpenACC::name() {
  return Impl::OpenACCInternal::impl_singleton()->name();
}
void OpenACC::print_configuration(std::ostream& os, bool verbose) const {
  m_space_instance->print_configuration(os, verbose);
}

uint32_t OpenACC::impl_instance_id() const noexcept {
  return m_space_instance->impl_get_instance_id();
}

int OpenACC::concurrency() {
  return Impl::OpenACCInternal::impl_singleton()->concurrency();
}

void OpenACC::fence(const std::string& name) const {
  Impl::OpenACCInternal::impl_singleton()->fence(name);
}
void OpenACC::impl_static_fence(const std::string& name) {
  Impl::OpenACCInternal::impl_singleton()->fence(
      name, Kokkos::Experimental::Impl::openacc_fence_is_static::yes);
}

void OpenACC::impl_initialize(InitializationSettings const&) {
  Impl::OpenACCInternal::impl_singleton()->impl_initialize();
}
void OpenACC::impl_finalize() {
  Impl::OpenACCInternal::impl_singleton()->impl_finalize();
}
int OpenACC::impl_is_initialized() {
  return Impl::OpenACCInternal::impl_singleton()->impl_is_initialized();
}
}  // Namespace Experimental

namespace Impl {
int g_openacc_space_factory_initialized =
    initialize_space_factory<Experimental::OpenACC>("170_OpenACC");
}  // namespace Impl
}  // Namespace Kokkos
