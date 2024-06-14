/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>

#include <string>
#include <vector>

namespace momentum {

// loads jointParameters stored in a binary format
// this format is used in a DenseRaC processing, for example
std::vector<JointParameters> loadJointParamsBinary(const std::string& filename);

} // namespace momentum
