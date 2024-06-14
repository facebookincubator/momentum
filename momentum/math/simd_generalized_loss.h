/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/generalized_loss.h>
#include <momentum/simd/simd.h>

namespace momentum {

/// SIMD version of the generalized loss function.
template <typename T>
class SimdGeneralizedLossT : public GeneralizedLossT<T> {
 public:
  using Base = GeneralizedLossT<T>;

  explicit SimdGeneralizedLossT(const T& a = Base::kL2, const T& c = T(1));
  [[nodiscard]] Packet<T> value(const Packet<T>& sqrError) const;
  [[nodiscard]] Packet<T> deriv(const Packet<T>& sqrError) const;
};

} // namespace momentum
