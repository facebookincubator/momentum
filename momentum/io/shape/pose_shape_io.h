/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/pose_shape.h>

namespace momentum {

PoseShape loadPoseShape(const std::string& filename, const Character& character);

} // namespace momentum
