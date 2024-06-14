/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Geometry>

#include "axel/math/RayTriangleIntersection.h"

namespace axel {

template <class T>
void getMinMaxBoundSegment(
    const Eigen::Vector3<T>& p0,
    const Eigen::Vector3<T>& p1,
    Eigen::Vector3<T>& minBound,
    Eigen::Vector3<T>& maxBound,
    T epsilon) {
  minBound.array() = p0.array().min(p1.array()) - epsilon;
  maxBound.array() = p0.array().max(p1.array()) + epsilon;
}

template <class T>
void getMinMaxBoundTriangle(
    const Eigen::Vector3<T>& p0,
    const Eigen::Vector3<T>& p1,
    const Eigen::Vector3<T>& p2,
    Eigen::Vector3<T>& minBound,
    Eigen::Vector3<T>& maxBound,
    T epsilon) {
  minBound.array() = p0.array().min(p1.array().min(p2.array())) - epsilon;
  maxBound.array() = p0.array().max(p1.array().max(p2.array())) + epsilon;
}

template <class T>
void getAdaptiveMinMaxBoundTriangle(
    const Eigen::Vector3<T>& p0,
    const Eigen::Vector3<T>& p1,
    const Eigen::Vector3<T>& p2,
    Eigen::Vector3<T>& minBound,
    Eigen::Vector3<T>& maxBound,
    T multiplicativeFactor = 0.01) {
  const Eigen::Array3<T> minArray = p0.array().min(p1.array().min(p2.array()));
  const Eigen::Array3<T> maxArray = p0.array().max(p1.array().max(p2.array()));
  const Eigen::Array3<T> epsilonArray = (maxArray - minArray) * multiplicativeFactor;
  minBound.array() = minArray - epsilonArray;
  maxBound.array() = maxArray + epsilonArray;
}

} // namespace axel
