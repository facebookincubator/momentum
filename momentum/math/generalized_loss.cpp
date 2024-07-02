/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/generalized_loss.h"

#include "momentum/common/checks.h"
#include "momentum/common/log.h"

#include <cmath>

namespace momentum {

namespace {

// No 0.5 factor so it can be reused.
template <typename T>
T L2Loss(const T& sqrError, const T& invC2) {
  return sqrError * invC2;
}

template <typename T>
T derivL2Loss(const T& /* sqrError */, const T& invC2) {
  return invC2;
}

template <typename T>
T L1Loss(const T& sqrError, const T& invC2) {
  // decided not to skip the -1 so the value itself is meaningful
  return std::sqrt(L2Loss(sqrError, invC2) + T(1)) - T(1);
}

template <typename T>
T derivL1Loss(const T& sqrError, const T& invC2) {
  return T(0.5) * invC2 / std::sqrt(L2Loss(sqrError, invC2) + T(1));
}

template <typename T>
T CauchyLoss(const T& sqrError, const T& invC2) {
  return std::log(T(0.5) * L2Loss(sqrError, invC2) + T(1));
}

template <typename T>
T derivCauchyLoss(const T& sqrError, const T& invC2) {
  return invC2 / (invC2 * sqrError + T(2));
}

template <typename T>
T WelschLoss(const T& sqrError, const T& invC2) {
  return T(1) - std::exp(T(-0.5) * L2Loss(sqrError, invC2));
}

template <typename T>
T derivWelschLoss(const T& sqrError, const T& invC2) {
  return T(0.5) * invC2 * std::exp(T(-0.5) * L2Loss(sqrError, invC2));
}

} // namespace

template <typename T>
GeneralizedLossT<T>::GeneralizedLossT(const T& a, const T& c) : alpha_(a), invC2_(T(1) / (c * c)) {
  MT_CHECK(c > 0, "Parameter c should be positive but received {}", c);

  // Use a threshold around special values for both stability and speed
  if (alpha_ >= kL2 - kEps && alpha_ <= kL2 + kEps) {
    lossType_ = LossType::L2;
  } else if (alpha_ >= kL1 - kEps && alpha_ <= kL1 + kEps) {
    lossType_ = LossType::L1;
  } else if (alpha_ >= kCauchy - kEps && alpha_ <= kCauchy + kEps) {
    lossType_ = LossType::Cauchy;
  } else if (alpha_ <= kWelsch) {
    lossType_ = LossType::Welsch;
  } else {
    lossType_ = LossType::General;
  }
}

template <typename T>
T GeneralizedLossT<T>::value(const T& sqrError) const {
  switch (lossType_) {
    case LossType::L2:
      return L2Loss(sqrError, invC2_);
    case LossType::L1:
      return L1Loss(sqrError, invC2_);
    case LossType::Cauchy:
      return CauchyLoss(sqrError, invC2_);
    case LossType::Welsch:
      return WelschLoss(sqrError, invC2_);
    case LossType::General:
      // General case, slower
      return (std::pow(L2Loss(sqrError, invC2_) / std::abs(alpha_ - T(2)) + T(1), T(0.5) * alpha_) -
              T(1)) *
          std::abs(alpha_ - T(2)) / alpha_;
    default: {
      MT_LOGE("Invalid lossType_ ({})", static_cast<int>(lossType_));
      return {};
    }
  }
}

template <typename T>
T GeneralizedLossT<T>::deriv(const T& sqrError) const {
  switch (lossType_) {
    case LossType::L2:
      return derivL2Loss(sqrError, invC2_);
    case LossType::L1:
      return derivL1Loss(sqrError, invC2_);
    case LossType::Cauchy:
      return derivCauchyLoss(sqrError, invC2_);
    case LossType::Welsch:
      return derivWelschLoss(sqrError, invC2_);
    case LossType::General:
      // General case, slower
      return T(0.5) * invC2_ *
          std::pow(
                 L2Loss(sqrError, invC2_) / std::abs(alpha_ - T(2)) + T(1), T(0.5) * alpha_ - T(1));
    default: {
      MT_LOGE("Invalid lossType_ ({})", static_cast<int>(lossType_));
      return {};
    }
  }
}

template class GeneralizedLossT<float>;
template class GeneralizedLossT<double>;

} // namespace momentum
