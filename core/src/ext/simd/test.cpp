/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#include <iostream>
#include <iomanip>

#include "simd.hpp"

int main() {
  simd::simd_storage<double, simd::simd_abi::native> aa;
  simd::simd_storage<double, simd::simd_abi::native> ab;
  simd::simd_storage<double, simd::simd_abi::native> ac;
  simd::simd_storage<double, simd::simd_abi::native> ad;
  for (int i = 0; i < simd::simd<double, simd::simd_abi::native>::size(); ++i) {
    aa[i] = 1.0 * (i + 1);
    ab[i] = 0.5 * (i + 1);
    ac[i] = 0.1 * (i + 1);
    ad[i] = 0.0;
  }
  simd::simd<double, simd::simd_abi::native> sa;
  simd::simd<double, simd::simd_abi::native> sb;
  simd::simd<double, simd::simd_abi::native> sc;
  simd::simd<double, simd::simd_abi::native> sd;
  sa = aa;
  sb = ab;
  sc = ac;
  simd::simd_mask<double, simd::simd_abi::native> ma(false);
  sd = simd::choose(ma, cbrt(sa), fma(sa, sa, sc));
  ad = sd;
  std::cout << std::setprecision(6);
  for (int i = 0; i < simd::simd<double, simd::simd_abi::native>::size(); ++i) {
    std::cout << ad[i] << '\n';
  }
}
