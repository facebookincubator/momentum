/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/skeleton/parameters_io.h"

#include "momentum/character/character.h"
#include "momentum/io/skeleton/parameter_limits_io.h"
#include "momentum/io/skeleton/parameter_transform_io.h"

namespace momentum {

void loadParameters(std::unordered_map<std::string, std::string>& param, Character& character) {
  if (param.empty()) {
    // create an identity parameter transform
    character.initParameterTransform();
    return;
  }

  character.parameterTransform =
      parseParameterTransform(param["ParameterTransform"], character.skeleton);
  character.parameterTransform.parameterSets =
      parseParameterSets(param["ParameterSets"], character.parameterTransform);
  character.parameterTransform.poseConstraints =
      parsePoseConstraints(param["PoseConstraints"], character.parameterTransform);
  character.parameterLimits =
      parseParameterLimits(param["Limits"], character.skeleton, character.parameterTransform);
}

} // namespace momentum
