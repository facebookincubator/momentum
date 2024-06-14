/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "axel/BvhBase.h"

namespace axel {

/**
 * @brief Creates a bvh that speeds up spatial look-up for collision
 * detection.
 * @param type Type of the acceleration structure. Supported values are:
 * - "default": instantiates an implementation-specific default BVH
 * - "embree": instantiates an Embree-based BVH on supported (desktop) platforms. Otherwise,
 * creates a "default" BVH.
 * - "fast-bvh": instantiates a BVH based on the https://github.com/brandonpelfrey/Fast-BVH.
 * @param threadCount Specify how many threads to use for a given acceleration structure if
 * supported.
 * @return A unique pointer to the created acceleration structure.
 */
std::unique_ptr<BvhBased> createBvh(
    std::string_view type = "default",
    std::optional<uint32_t> threadCount = std::nullopt);

} // namespace axel
