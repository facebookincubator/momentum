/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "axel/common/Types.h"

namespace axel {

template <typename ScalarType>
struct BoundingBox {
  using Scalar = ScalarType;

  BoundingBox() = default;
  BoundingBox(const Eigen::Vector3<Scalar>& min, const Eigen::Vector3<Scalar>& max, Index id = 0);
  explicit BoundingBox(const Eigen::Vector3<Scalar>& p, Scalar thickness = 0.0);

  const Eigen::Vector3<Scalar>& min() const;
  const Eigen::Vector3<Scalar>& max() const;
  Eigen::Vector3<Scalar> center() const;

  /// Returns the squared volume of the bounding box.
  [[nodiscard]] Scalar squaredVolume() const;

  void extend(const Eigen::Vector3<Scalar>& p);
  void extend(const BoundingBox& b);

  bool intersects(const Eigen::Vector3<Scalar>& origin, const Eigen::Vector3<Scalar>& direction)
      const;
  bool intersects(const BoundingBox& box) const;

  bool intersectsBranchless(
      const Eigen::Vector3<Scalar>& origin,
      const Eigen::Vector3<Scalar>& direction) const;

  [[nodiscard]] bool contains(const Eigen::Vector3<Scalar>& point) const;
  [[nodiscard]] bool contains(const BoundingBox& other) const;

  Index maxDimension() const;

  Eigen::AlignedBox<Scalar, 3> aabb;
  Index id{0};
};

using BoundingBoxf = BoundingBox<float>;
using BoundingBoxd = BoundingBox<double>;

extern template struct BoundingBox<float>;
extern template struct BoundingBox<double>;

} // namespace axel
