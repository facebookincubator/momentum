/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <drjit/packet.h>

namespace axel {

// DrJit's DefaultSize constant is based on single precision float (4 for SSE, 8 for AVX, 16 for
// AVX512).
// If we are dealing with a platform that doesn't support SIMD/isn't detected, we'll use the
// fallback sizes that will compile down to scalar loops.

template <typename ScalarType>
struct NativeLaneWidth {
  static constexpr size_t value = 0;
};

template <>
struct NativeLaneWidth<float> {
  static constexpr size_t value = drjit::DefaultSize <= 1 ? 4 : drjit::DefaultSize;
};

template <>
struct NativeLaneWidth<int32_t> {
  static constexpr size_t value = drjit::DefaultSize <= 1 ? 4 : drjit::DefaultSize;
};

template <>
struct NativeLaneWidth<double> {
  static constexpr size_t value = drjit::DefaultSize <= 1 ? 2 : drjit::DefaultSize / 2;
};

template <typename ScalarType>
inline static constexpr size_t kNativeLaneWidth = NativeLaneWidth<ScalarType>::value;

template <typename Scalar, size_t LaneWidth = kNativeLaneWidth<Scalar>>
using WideScalar = drjit::Packet<Scalar, LaneWidth>;

template <typename Scalar, size_t LaneWidth = kNativeLaneWidth<Scalar>>
using WideVec3 = drjit::Array<WideScalar<Scalar, LaneWidth>, 3>; // Each element is a lane.

// Memory layout of a boolean SIMD mask is ISA dependent.
// We need to retrieve the correct type based on underlying native support.
template <typename VectorType>
using WideMask = drjit::mask_t<VectorType>;

using WideScalarf = WideScalar<float>;
using WideScalard = WideScalar<double>;
using WideVec3f = WideVec3<float>;
using WideVec3d = WideVec3<double>;

} // namespace axel
