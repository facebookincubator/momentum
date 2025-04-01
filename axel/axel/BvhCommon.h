/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

#include <Eigen/Core>

namespace axel {

template <typename S>
struct IntersectionResult {
  using Scalar = S;
  int32_t triangleId{0};
  S hitDistance{std::numeric_limits<S>::max()};
  Eigen::Vector3<S> hitPoint;
  Eigen::Vector3<S> baryCoords;
};

using IntersectionResultf = IntersectionResult<float>;
using IntersectionResultd = IntersectionResult<double>;

// TODO(nemanjab): Fix-it: unify index types.
inline constexpr uint32_t kInvalidTriangleIdx = std::numeric_limits<uint32_t>::max();

template <typename S>
struct ClosestSurfacePointResult {
  Eigen::Vector3<S> point;
  uint32_t triangleIdx = kInvalidTriangleIdx;

  // TODO: Remove optional once all the queries support barycentric coordinates.
  std::optional<Eigen::Vector3<S>> baryCoords;
};

using ClosestSurfacePointResultf = ClosestSurfacePointResult<float>;
using ClosestSurfacePointResultd = ClosestSurfacePointResult<double>;

} // namespace axel
