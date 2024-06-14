/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_limits.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>
#include <Eigen/Core>

#include <random>
#include <string>

namespace momentum {

// Returns the error as a fraction of the target value.
// Uses max(targetValue, 1) in the denominator to avoid
// overflow (and to deal with error in the target).
template <typename T>
T relativeError(T targetValue, T measuredValue, T minScale = T(1)) {
  return std::abs(measuredValue - targetValue) /
      std::max(minScale, std::max(std::abs(measuredValue), std::abs(targetValue)));
}

ModelParameters randomBodyParameters(
    const ParameterTransform& bodyParamTransform,
    std::mt19937& rng);

Eigen::VectorXf randomVec(std::mt19937& rng, int sz);

} // namespace momentum
