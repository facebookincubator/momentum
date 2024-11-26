/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_limits.h>

namespace momentum {

ParameterLimits parseParameterLimits(
    const std::string& data,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform);

std::string writeParameterLimits(
    const ParameterLimits& parameterLimits,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform);

} // namespace momentum
