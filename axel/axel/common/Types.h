/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace axel {

using Index = int32_t;
static_assert(sizeof(Index) == 4);
inline constexpr Index kInvalidIndex = Index{-1};

using Size = std::ptrdiff_t;

} // namespace axel
