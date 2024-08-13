/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test_util.h"

#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton.h>

namespace momentum {

using namespace Eigen;

ModelParameters randomBodyParameters(
    const ParameterTransform& bodyParamTransform,
    std::mt19937& rng) {
  std::uniform_real_distribution<float> unifRot(-pi() / 4.0, pi() / 4.0);
  std::uniform_real_distribution<float> unifTrans(-5.0, 5.0);
  std::uniform_real_distribution<float> unifScale(-0.5, 0.5);

  const size_t nBodyParams = bodyParamTransform.numAllModelParameters();
  ModelParameters result(nBodyParams);
  for (size_t iParam = 0; iParam < nBodyParams; ++iParam) {
    const auto& name = bodyParamTransform.name[iParam];
    if (name.find("scale_") != std::string::npos) {
      result[iParam] = unifScale(rng);
    } else if (
        name.find("_tx") != std::string::npos || name.find("_ty") != std::string::npos ||
        name.find("_tz") != std::string::npos) {
      result[iParam] = unifTrans(rng);
    } else {
      // Assume it's a rotation parameter
      result[iParam] = unifRot(rng);
    }
  }

  return result;
}

Eigen::VectorXf randomVec(std::mt19937& rng, int sz) {
  std::normal_distribution<float> dist;

  Eigen::VectorXf result(sz);
  for (int j = 0; j < sz; ++j) {
    result(j) = dist(rng);
  }
  return result;
}

} // namespace momentum
