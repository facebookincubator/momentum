/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/simd_generalized_loss.h"

#include "momentum/common/log.h"

#include <drjit/math.h>

#include <cmath>

namespace momentum {

namespace {

// No 0.5 factor so it can be reused.
template <typename T>
Packet<T> L2Loss(const Packet<T>& sqrError, T invC2) {
  return sqrError * invC2;
}

template <typename T>
Packet<T> derivL2Loss(const Packet<T>& /* sqrError */, T invC2) {
  return invC2;
}

template <typename T>
Packet<T> L1Loss(const Packet<T>& sqrError, T invC2) {
  // decided not to skip the -1 so the value itself is meaningful
  return drjit::sqrt(L2Loss(sqrError, invC2) + T(1)) - T(1);
}

template <typename T>
Packet<T> derivL1Loss(const Packet<T>& sqrError, T invC2) {
  return T(0.5) * invC2 / drjit::sqrt(L2Loss(sqrError, invC2) + T(1));
}

template <typename T>
Packet<T> CauchyLoss(const Packet<T>& sqrError, T invC2) {
  return drjit::log(T(0.5) * L2Loss(sqrError, invC2) + T(1));
}

template <typename T>
Packet<T> derivCauchyLoss(const Packet<T>& sqrError, T invC2) {
  return invC2 / (invC2 * sqrError + T(2));
}

template <typename T>
Packet<T> WelschLoss(const Packet<T>& sqrError, T invC2) {
  return T(1) - drjit::exp(T(-0.5) * L2Loss(sqrError, invC2));
}

template <typename T>
Packet<T> derivWelschLoss(const Packet<T>& sqrError, T invC2) {
  return (T(0.5) * invC2) * drjit::exp(T(-0.5) * L2Loss(sqrError, invC2));
}

} // namespace

template <typename T>
SimdGeneralizedLossT<T>::SimdGeneralizedLossT(const T& a, const T& c) : Base(a, c) {
  // Empty
}

template <typename T>
Packet<T> SimdGeneralizedLossT<T>::value(const Packet<T>& sqrError) const {
  switch (this->lossType_) {
    case Base::LossType::L2:
      return L2Loss(sqrError, this->invC2_);
    case Base::LossType::L1:
      return L1Loss(sqrError, this->invC2_);
    case Base::LossType::Cauchy:
      return CauchyLoss(sqrError, this->invC2_);
    case Base::LossType::Welsch:
      return WelschLoss(sqrError, this->invC2_);
    case Base::LossType::General:
      // General case, slower
      return (drjit::pow(
                  L2Loss(sqrError, this->invC2_) / std::fabs(this->alpha_ - T(2)) + T(1),
                  T(0.5) * this->alpha_) -
              T(1)) *
          (std::fabs(this->alpha_ - T(2)) / this->alpha_);
    default: {
      MT_LOGE("Invalid lossType_ ({})", static_cast<int>(this->lossType_));
      return {};
    }
  }
}

template <typename T>
Packet<T> SimdGeneralizedLossT<T>::deriv(const Packet<T>& sqrError) const {
  switch (this->lossType_) {
    case Base::LossType::L2:
      return derivL2Loss(sqrError, this->invC2_);
    case Base::LossType::L1:
      return derivL1Loss(sqrError, this->invC2_);
    case Base::LossType::Cauchy:
      return derivCauchyLoss(sqrError, this->invC2_);
    case Base::LossType::Welsch:
      return derivWelschLoss(sqrError, this->invC2_);
    case Base::LossType::General:
      // General case, slower
      return (T(0.5) * this->invC2_) *
          drjit::pow(
                 L2Loss(sqrError, this->invC2_) / std::fabs(this->alpha_ - T(2)) + T(1),
                 T(0.5) * this->alpha_ - T(1));
    default: {
      MT_LOGE("Invalid lossType_ ({})", static_cast<int>(this->lossType_));
      return {};
    }
  }
}

template class SimdGeneralizedLossT<float>;
template class SimdGeneralizedLossT<double>;

} // namespace momentum
