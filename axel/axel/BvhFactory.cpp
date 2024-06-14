/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/BvhFactory.h"

#define DEFAULT_LOG_CHANNEL "AXEL: BvhFactory"
#include "axel/Log.h"

#include "axel/Bvh.h"

#ifdef AXEL_BVH_EMBREE
#include "axel/BvhEmbree.h"
#endif

namespace axel {

std::unique_ptr<BvhBased> createBvh(
    const std::string_view type,
    const std::optional<uint32_t> threadCount) {
  if (type == "fast-bvh") {
    return std::make_unique<Bvhd>();
  }

  if (type == "embree") {
#ifdef AXEL_BVH_EMBREE
    return std::make_unique<BVHEmbree>(threadCount);
#else
    XR_LOGW("Embree is not supported on this platform! Creating default BVH instead.");
    return std::make_unique<Bvhd>();
#endif
  }

#ifdef AXEL_BVH_EMBREE
  return std::make_unique<BVHEmbree>(threadCount);
#else
  return std::make_unique<Bvhd>();
#endif
}

} // namespace axel
