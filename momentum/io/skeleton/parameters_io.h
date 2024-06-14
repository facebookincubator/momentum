/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>

#include <string>
#include <unordered_map>

namespace momentum {

// Load the input parameters onto the character
void loadParameters(std::unordered_map<std::string, std::string>& param, Character& character);

} // namespace momentum
