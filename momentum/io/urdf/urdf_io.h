/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/common/filesystem.h>
#include <momentum/math/types.h>

#include <gsl/span>

namespace momentum {

template <typename T = float>
[[nodiscard]] SkeletonT<T> loadUrdfSkeleton(const filesystem::path& filepath);

template <typename T = float>
[[nodiscard]] CharacterT<T> loadUrdfCharacter(const filesystem::path& filepath);

} // namespace momentum
