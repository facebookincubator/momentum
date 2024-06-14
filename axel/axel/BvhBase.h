/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <vector>

#include "axel/BoundingBox.h"
#include "axel/common/Types.h"

namespace axel {

template <typename S>
class BvhBase {
 public:
  using Scalar = S;
  using QueryBuffer = std::array<uint32_t, 512>;
  BvhBase() = default;
  virtual ~BvhBase() = default;

  BvhBase(const BvhBase&) = default;
  BvhBase& operator=(const BvhBase&) = default;

  BvhBase(BvhBase&&) noexcept = default;
  BvhBase& operator=(BvhBase&&) noexcept = default;

  /**
   * @brief Rebuilds the bvh with a new vector of bounding boxes.
   * @param bboxes The new bounding boxes.
   */
  virtual void setBoundingBoxes(const std::vector<BoundingBox<S>>& bboxes) = 0;

  /**
   * @brief Checks for an intersection between a given bounding box and the bvh.
   * All IDs of the primitives that overlap with the given bounding box will be returned
   * in a dynamically-allocated vector.
   * @param box The input bounding box to intersect with the bvh.
   * @return A vector of primitive IDs that intersect with the box.
   */
  [[nodiscard]] virtual std::vector<uint32_t> query(const BoundingBox<S>& box) const = 0;

  /**
   * @brief Checks for an intersection between a given bounding box and the bvh.
   * The query permits only up to first 512 intersections, which is the limit
   * of the query buffer. In practice this shouldn't make a difference,
   * but performance-wise we don't allocate tens of thousands of vectors on the heap.
   * @param box The input bounding box to perform the query with.
   * @param hits The buffer where the hit primitive IDs will be stored.
   * @return The number of hit primitive IDs. When using the hits buffer afterwards,
   * hit primitives are within [0, hitCount - 1] interval of the hits buffer.
   */
  virtual uint32_t query(const BoundingBox<S>& box, QueryBuffer& hits) const = 0;

  /**
   * @brief Checks for an intersection between a ray and the bvh.
   * @param origin The origin of the ray in the same space as the bvh data.
   * @param direction The direction of the ray in the same space as the bvh data.
   * @return A vector of primitive IDs that intersect with the ray.
   */
  [[nodiscard]] virtual std::vector<uint32_t> query(
      const Eigen::Vector3<S>& origin,
      const Eigen::Vector3<S>& direction) const = 0;

  [[nodiscard]] virtual Size getPrimitiveCount() const = 0;
};

using BvhBasef = BvhBase<float>;
using BvhBased = BvhBase<double>;

} // namespace axel
