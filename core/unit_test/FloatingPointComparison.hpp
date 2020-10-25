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

#ifndef KOKKOS_FLOATING_POINT_COMPARISON_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Error.hpp>

//____________________________________________________________________________//
// FIXME remove definition below when available from Kokkos numeric traits
//____________________________________________________________________________//
#include <cfloat>
namespace Kokkos {
namespace Experimental {
// clang-format off
template <class> struct finite_min;
template <> struct finite_min<float>       { static constexpr       float value = -FLT_MAX; };
template <> struct finite_min<double>      { static constexpr      double value = -DBL_MAX; };
template <> struct finite_min<long double> { static constexpr long double value = -LDBL_MAX; };
template <class> struct finite_max;
template <> struct finite_max<float>       { static constexpr       float value = FLT_MAX; };
template <> struct finite_max<double>      { static constexpr      double value = DBL_MAX; };
template <> struct finite_max<long double> { static constexpr long double value = LDBL_MAX; };
// clang-format on
}  // namespace Experimental
}  // namespace Kokkos
//____________________________________________________________________________//

//____________________________________________________________________________//
//____________________________________________________________________________//
// Code below was taken from Boost.Test Version 1.74.0 and modified to include
// __host__ __device__ annotations and remove Boost dependencies.
//____________________________________________________________________________//
//____________________________________________________________________________//

// clang-format off

//  (C) Copyright Gennadiy Rozental 2001.
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org/libs/test for the library home page.
//
//!@file
//!@brief algorithms for comparing floating point values
// ***************************************************************************

namespace fpc {

// ************************************************************************** //
// **************                 fpc::strength                ************** //
// ************************************************************************** //

//! Method for comparing floating point numbers
enum strength {
    FPC_STRONG, //!< "Very close"   - equation 2' in docs, the default
    FPC_WEAK    //!< "Close enough" - equation 3' in docs.
};


// ************************************************************************** //
// **************         tolerance presentation types         ************** //
// ************************************************************************** //

template<typename FPT>
struct percent_tolerance_t {
    KOKKOS_FUNCTION
    explicit    percent_tolerance_t( FPT v ) : m_value( v ) {}

    FPT m_value;
};

//____________________________________________________________________________//

template<typename FPT>
KOKKOS_INLINE_FUNCTION percent_tolerance_t<FPT>
percent_tolerance( FPT v )
{
    return percent_tolerance_t<FPT>( v );
}

//____________________________________________________________________________//

// ************************************************************************** //
// **************                    details                   ************** //
// ************************************************************************** //

namespace fpc_detail {

// FPT is Floating-Point Type: float, double, long double or User-Defined.
template<typename FPT>
KOKKOS_INLINE_FUNCTION FPT
fpt_abs( FPT fpv )
{
    return fpv < static_cast<FPT>(0) ? -fpv : fpv;
}

//____________________________________________________________________________//

template<typename FPT>
struct fpt_limits
{
  KOKKOS_FUNCTION
  static FPT    min_value() { return Kokkos::Experimental::finite_min<FPT>::value; }
  KOKKOS_FUNCTION
  static FPT    max_value() { return Kokkos::Experimental::finite_max<FPT>::value; }
};

//____________________________________________________________________________//

// both f1 and f2 are unsigned here
template<typename FPT>
KOKKOS_INLINE_FUNCTION FPT
safe_fpt_division( FPT f1, FPT f2 )
{
    // Avoid overflow.
    if( (f2 < static_cast<FPT>(1))  && (f1 > f2*fpt_limits<FPT>::max_value()) )
        return fpt_limits<FPT>::max_value();

    // Avoid underflow.
    if( (fpt_abs(f1) <= fpt_limits<FPT>::min_value()) ||
        ((f2 > static_cast<FPT>(1)) && (f1 < f2*fpt_limits<FPT>::min_value())) )
        return static_cast<FPT>(0);

    return f1/f2;
}

//____________________________________________________________________________//

template<typename FPT, typename ToleranceType>
KOKKOS_INLINE_FUNCTION FPT
fraction_tolerance( ToleranceType tolerance )
{
  return static_cast<FPT>(tolerance);
}

//____________________________________________________________________________//

template<typename FPT2, typename FPT>
KOKKOS_INLINE_FUNCTION FPT2
fraction_tolerance( percent_tolerance_t<FPT> tolerance )
{
    return FPT2(tolerance.m_value)*FPT2(0.01);
}

//____________________________________________________________________________//

} // namespace fpc_detail

// ************************************************************************** //
// **************             close_at_tolerance               ************** //
// ************************************************************************** //


/*!@brief Predicate for comparing floating point numbers
 *
 * This predicate is used to compare floating point numbers. In addition the comparison produces maximum
 * related difference, which can be used to generate detailed error message
 * The methods for comparing floating points are detailed in the documentation. The method is chosen
 * by the @ref boost::math::fpc::strength given at construction.
 *
 * This predicate is not suitable for comparing to 0 or to infinity.
 */
template<typename FPT>
class close_at_tolerance {
public:
    // Public typedefs
    // NOLINTNEXTLINE(modernize-use-using)
    typedef bool result_type;

    // Constructor
    template<typename ToleranceType>
    KOKKOS_FUNCTION
    explicit    close_at_tolerance( ToleranceType tolerance, fpc::strength fpc_strength = FPC_STRONG )
    : m_fraction_tolerance( fpc_detail::fraction_tolerance<FPT>( tolerance ) )
    , m_strength( fpc_strength )
    , m_tested_rel_diff( 0 )
    {
        KOKKOS_ENSURES( m_fraction_tolerance >= FPT(0) && "tolerance must not be negative!" ); // no reason for tolerance to be negative
    }

    // Access methods
    //! Returns the tolerance
    KOKKOS_FUNCTION
    FPT                 fraction_tolerance() const  { return m_fraction_tolerance; }

    //! Returns the comparison method
    KOKKOS_FUNCTION
    fpc::strength       strength() const            { return m_strength; }

    //! Returns the failing fraction
    KOKKOS_FUNCTION
    FPT                 tested_rel_diff() const     { return m_tested_rel_diff; }

    /*! Compares two floating point numbers a and b such that their "left" relative difference |a-b|/a and/or
     * "right" relative difference |a-b|/b does not exceed specified relative (fraction) tolerance.
     *
     *  @param[in] left first floating point number to be compared
     *  @param[in] right second floating point number to be compared
     *
     * What is reported by @c tested_rel_diff in case of failure depends on the comparison method:
     * - for @c FPC_STRONG: the max of the two fractions
     * - for @c FPC_WEAK: the min of the two fractions
     * The rationale behind is to report the tolerance to set in order to make a test pass.
     */
    KOKKOS_FUNCTION
    bool                operator()( FPT left, FPT right ) const
    {
        FPT diff              = fpc_detail::fpt_abs<FPT>( left - right );
        FPT fraction_of_right = fpc_detail::safe_fpt_division( diff, fpc_detail::fpt_abs( right ) );
        FPT fraction_of_left  = fpc_detail::safe_fpt_division( diff, fpc_detail::fpt_abs( left ) );

        FPT max_rel_diff = fraction_of_left > fraction_of_right ? fraction_of_left : fraction_of_right;
        FPT min_rel_diff = fraction_of_left > fraction_of_right ? fraction_of_right : fraction_of_left;

        m_tested_rel_diff = m_strength == FPC_STRONG ? max_rel_diff : min_rel_diff;

        return m_tested_rel_diff <= m_fraction_tolerance;
    }

private:
    // Data members
    FPT                 m_fraction_tolerance;
    fpc::strength       m_strength;
    mutable FPT         m_tested_rel_diff;
};

// ************************************************************************** //
// **************            small_with_tolerance              ************** //
// ************************************************************************** //


/*!@brief Predicate for comparing floating point numbers against 0
 *
 * Serves the same purpose as boost::math::fpc::close_at_tolerance, but used when one
 * of the operand is null.
 */
template<typename FPT>
class small_with_tolerance {
public:
    // Public typedefs
    // NOLINTNEXTLINE(modernize-use-using)
    typedef bool result_type;

    // Constructor
    KOKKOS_FUNCTION
    explicit    small_with_tolerance( FPT tolerance ) // <= absolute tolerance
    : m_tolerance( tolerance )
    {
        KOKKOS_ENSURES( m_tolerance >= FPT(0) ); // no reason for the tolerance to be negative
    }

    // Action method
    KOKKOS_FUNCTION
    bool        operator()( FPT fpv ) const
    {
        return fpc::fpc_detail::fpt_abs( fpv ) <= m_tolerance;
    }

private:
    // Data members
    FPT         m_tolerance;
};

// ************************************************************************** //
// **************                  is_small                    ************** //
// ************************************************************************** //

template<typename FPT>
KOKKOS_INLINE_FUNCTION bool
is_small( FPT fpv, FPT tolerance )
{
    return small_with_tolerance<FPT>( tolerance )( fpv );
}

//____________________________________________________________________________//

} // namespace fpc

#endif
